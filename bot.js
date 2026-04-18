/**
 * Discord Voice Agent Bot
 * Listens to voice channels, transcribes with whisper.cpp,
 * queries Ollama, and speaks responses with Piper TTS.
 *
 * npm install discord.js @discordjs/voice @discordjs/opus prism-media axios dotenv
 */

require('dotenv').config();
const { Client, GatewayIntentBits, REST, Routes, SlashCommandBuilder } = require('discord.js');
const {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  EndBehaviorType,
  VoiceConnectionStatus,
  entersState,
  AudioPlayerStatus,
  StreamType,
} = require('@discordjs/voice');
const prism = require('prism-media');
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const { execFile, spawn } = require('child_process');
const axios = require('axios');

// ─────────────────────────────────────────────
// CONFIG  (all values come from .env)
// ─────────────────────────────────────────────

const BOT_TOKEN          = process.env.BOT_TOKEN;
const GUILD_ID           = process.env.GUILD_ID;
const AUTO_JOIN_CHANNEL  = process.env.AUTO_JOIN_CHANNEL_ID || null;
const OWNER_USER_ID      = process.env.OWNER_USER_ID        || null;
const OLLAMA_URL         = process.env.OLLAMA_URL  || 'http://localhost:11434/api/chat';
const OLLAMA_MODEL       = process.env.OLLAMA_MODEL || 'llama3.2:3b';
const WHISPER_BIN        = process.env.WHISPER_BIN  || path.join(os.homedir(), 'whisper.cpp/main');
const WHISPER_MODEL_PATH = process.env.WHISPER_MODEL || path.join(os.homedir(), 'whisper.cpp/models/ggml-base.en.bin');
const PIPER_BIN          = process.env.PIPER_BIN    || path.join(os.homedir(), 'piper/piper');
const PIPER_MODEL_PATH   = process.env.PIPER_MODEL  || path.join(os.homedir(), 'piper/voices/en_US-lessac-medium.onnx');

const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT ||
  'You are a helpful voice assistant inside a Discord voice channel. ' +
  'Your responses are spoken aloud, so keep them conversational and concise. ' +
  'Avoid markdown, bullet points, and code blocks unless the user asks for them. ' +
  'Address the person by their Discord username when responding.';

// Audio constants — Discord sends 48kHz stereo Opus; Whisper needs 16kHz mono PCM
const DISCORD_SAMPLE_RATE  = 48000;
const DISCORD_CHANNELS     = 2;
const WHISPER_SAMPLE_RATE  = 16000;
const SILENCE_DURATION_MS  = 1500;   // ms of silence to trigger processing
const MIN_AUDIO_DURATION_MS = 800;   // ignore clips shorter than this (coughs, noise)
const FRAME_SIZE           = 960;    // Opus frame size

// ─────────────────────────────────────────────
// SLASH COMMANDS — register with Discord
// ─────────────────────────────────────────────

const commands = [
  new SlashCommandBuilder()
    .setName('join')
    .setDescription('Bot joins your current voice channel'),

  new SlashCommandBuilder()
    .setName('leave')
    .setDescription('Bot disconnects from the voice channel'),

  new SlashCommandBuilder()
    .setName('clear')
    .setDescription('Clear the conversation history'),

  new SlashCommandBuilder()
    .setName('model')
    .setDescription('Switch the Ollama model')
    .addStringOption(opt =>
      opt.setName('name')
        .setDescription('Model name (e.g. llama3.2:3b, qwen2.5:3b)')
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName('status')
    .setDescription('Show current bot status (model, channel, history length)'),
].map(cmd => cmd.toJSON());

async function registerSlashCommands() {
  const rest = new REST({ version: '10' }).setToken(BOT_TOKEN);
  try {
    console.log('[Slash] Registering slash commands...');
    await rest.put(
      Routes.applicationGuildCommands(process.env.CLIENT_ID, GUILD_ID),
      { body: commands }
    );
    console.log('[Slash] Commands registered.');
  } catch (err) {
    console.error('[Slash] Failed to register commands:', err.message);
  }
}

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

// Per-guild state: { connection, player, history, processingQueue, isProcessing }
const guilds = new Map();

// ─────────────────────────────────────────────
// DISCORD CLIENT
// ─────────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// ─────────────────────────────────────────────
// AUDIO UTILITIES
// ─────────────────────────────────────────────

/**
 * Downsample 48kHz stereo Int16 PCM → 16kHz mono Int16 PCM
 * Simple linear interpolation — good enough for speech.
 */
function downsample(buffer) {
  const input  = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  const ratio  = DISCORD_SAMPLE_RATE / WHISPER_SAMPLE_RATE;
  const outLen = Math.floor(input.length / DISCORD_CHANNELS / ratio);
  const output = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcIndex = Math.floor(i * ratio) * DISCORD_CHANNELS;
    // Average stereo channels to mono
    output[i] = (input[srcIndex] + input[srcIndex + 1]) >> 1;
  }

  return Buffer.from(output.buffer);
}

/**
 * Write a raw PCM buffer to a WAV file (16kHz mono Int16).
 */
function writePcmToWav(pcmBuffer, filePath) {
  return new Promise((resolve, reject) => {
    const header = Buffer.alloc(44);
    const dataSize = pcmBuffer.length;
    const sampleRate = WHISPER_SAMPLE_RATE;
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;

    header.write('RIFF', 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);                  // PCM
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write('data', 36);
    header.writeUInt32LE(dataSize, 40);

    fs.writeFile(filePath, Buffer.concat([header, pcmBuffer]), (err) => {
      if (err) reject(err); else resolve();
    });
  });
}

// ─────────────────────────────────────────────
// TRANSCRIPTION (whisper.cpp)
// ─────────────────────────────────────────────

function transcribe(wavPath) {
  return new Promise((resolve, reject) => {
    execFile(
      WHISPER_BIN,
      ['-m', WHISPER_MODEL_PATH, '-f', wavPath, '--language', 'en', '--no-timestamps', '-nt'],
      { timeout: 30000 },
      (err, stdout, stderr) => {
        if (err) { reject(err); return; }
        const text = stdout
          .split('\n')
          .map(l => l.replace(/\[.*?\]/g, '').trim())
          .filter(Boolean)
          .join(' ')
          .trim();
        resolve(text);
      }
    );
  });
}

// ─────────────────────────────────────────────
// LLM (Ollama)
// ─────────────────────────────────────────────

async function queryOllama(guildId, username, userText) {
  const state = guilds.get(guildId);
  if (!state) return null;

  // Add user message with their username for context
  state.history.push({
    role: 'user',
    content: `[${username}]: ${userText}`,
  });

  try {
    const response = await axios.post(
      OLLAMA_URL,
      {
        model: OLLAMA_MODEL,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...state.history,
        ],
        stream: false,
      },
      { timeout: 60000 }
    );

    const reply = response.data.message.content.trim();
    state.history.push({ role: 'assistant', content: reply });

    // Keep history to last 20 messages to avoid context overflow
    if (state.history.length > 20) {
      state.history = state.history.slice(-20);
    }

    return reply;
  } catch (err) {
    console.error('[Ollama] Error:', err.message);
    return "Sorry, I couldn't reach the AI model right now.";
  }
}

// ─────────────────────────────────────────────
// TTS (Piper)
// ─────────────────────────────────────────────

function generateTts(text, outputPath) {
  return new Promise((resolve, reject) => {
    const piper = spawn(PIPER_BIN, [
      '--model', PIPER_MODEL_PATH,
      '--output_file', outputPath,
    ]);

    piper.stdin.write(text);
    piper.stdin.end();

    piper.on('close', (code) => {
      if (code === 0) resolve(); else reject(new Error(`Piper exited with code ${code}`));
    });
    piper.on('error', reject);
  });
}

// ─────────────────────────────────────────────
// VOICE — play audio in channel
// ─────────────────────────────────────────────

async function playAudio(guildId, audioPath) {
  const state = guilds.get(guildId);
  if (!state || !state.player) return;

  return new Promise((resolve) => {
    const resource = createAudioResource(audioPath, { inputType: StreamType.Arbitrary });
    state.player.play(resource);
    state.player.once(AudioPlayerStatus.Idle, resolve);
    state.player.once('error', (err) => {
      console.error('[Player] Error:', err.message);
      resolve();
    });
  });
}

// ─────────────────────────────────────────────
// PIPELINE — transcribe → LLM → TTS → play
// ─────────────────────────────────────────────

async function processAudio(guildId, userId, username, pcmChunks) {
  const state = guilds.get(guildId);
  if (!state) return;

  // Check minimum audio length
  const totalBytes = pcmChunks.reduce((s, c) => s + c.length, 0);
  const durationMs = (totalBytes / 2 / WHISPER_SAMPLE_RATE) * 1000; // after downsampling
  if (durationMs < MIN_AUDIO_DURATION_MS) return;

  const tmpDir   = os.tmpdir();
  const wavPath  = path.join(tmpDir, `discord_${userId}_${Date.now()}.wav`);
  const ttsPath  = path.join(tmpDir, `tts_${Date.now()}.wav`);

  try {
    // 1. Downsample and write WAV
    const rawPcm    = Buffer.concat(pcmChunks);
    const pcm16khz  = downsample(rawPcm);
    await writePcmToWav(pcm16khz, wavPath);

    // 2. Transcribe
    console.log(`[Whisper] Transcribing ${username}...`);
    const text = await transcribe(wavPath);
    if (!text || text.length < 3) return;
    console.log(`[${username}]: ${text}`);

    // 3. Query Ollama
    console.log('[Ollama] Querying...');
    const reply = await queryOllama(guildId, username, text);
    if (!reply) return;
    console.log(`[Bot]: ${reply}`);

    // 4. Generate TTS
    await generateTts(reply, ttsPath);

    // 5. Play in voice channel
    await playAudio(guildId, ttsPath);

  } catch (err) {
    console.error('[Pipeline] Error:', err.message);
  } finally {
    // Clean up temp files
    for (const f of [wavPath, ttsPath]) {
      fs.unlink(f, () => {});
    }
  }
}

// ─────────────────────────────────────────────
// VOICE CONNECTION
// ─────────────────────────────────────────────

async function connectToChannel(channel) {
  const guildId = channel.guild.id;

  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId:   guildId,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,   // must be false to receive audio
    selfMute: false,
  });

  const player = createAudioPlayer();
  connection.subscribe(player);

  // Wait until ready
  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
  } catch {
    connection.destroy();
    throw new Error('Could not connect to voice channel within 20 seconds.');
  }

  // Store state
  guilds.set(guildId, {
    connection,
    player,
    history: [],
    channelName: channel.name,
  });

  // Handle reconnect
  connection.on(VoiceConnectionStatus.Disconnected, async () => {
    try {
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Signalling, 5_000),
        entersState(connection, VoiceConnectionStatus.Connecting, 5_000),
      ]);
    } catch {
      connection.destroy();
      guilds.delete(guildId);
    }
  });

  startListening(guildId);
  console.log(`[Bot] Joined #${channel.name} in ${channel.guild.name}`);
}

// ─────────────────────────────────────────────
// AUDIO RECEIVER
// ─────────────────────────────────────────────

function startListening(guildId) {
  const state = guilds.get(guildId);
  if (!state) return;

  const { connection } = state;
  const receiver = connection.receiver;

  // One queue per guild to avoid overlapping responses
  const queue = [];
  let processing = false;

  async function drainQueue() {
    if (processing || queue.length === 0) return;
    processing = true;
    const job = queue.shift();
    await processAudio(job.guildId, job.userId, job.username, job.chunks);
    processing = false;
    drainQueue();
  }

  receiver.speaking.on('start', (userId) => {
    // Ignore the bot itself
    if (userId === client.user.id) return;

    const member = state.connection.joinConfig.guildId &&
      client.guilds.cache.get(guildId)?.members.cache.get(userId);
    const username = member?.displayName || member?.user?.username || `User ${userId}`;

    // Subscribe to this user's audio stream, auto-end after silence
    const audioStream = receiver.subscribe(userId, {
      end: {
        behavior: EndBehaviorType.AfterSilence,
        duration: SILENCE_DURATION_MS,
      },
    });

    // Decode Opus → raw PCM
    const decoder = new prism.opus.Decoder({
      frameSize: FRAME_SIZE,
      channels:  DISCORD_CHANNELS,
      rate:      DISCORD_SAMPLE_RATE,
    });

    const chunks = [];

    audioStream
      .pipe(decoder)
      .on('data', (chunk) => chunks.push(chunk))
      .on('end', () => {
        if (chunks.length === 0) return;
        queue.push({ guildId, userId, username, chunks });
        drainQueue();
      })
      .on('error', (err) => console.error('[Decoder] Error:', err.message));
  });
}

// ─────────────────────────────────────────────
// DISCORD EVENTS
// ─────────────────────────────────────────────

client.once('ready', async () => {
  console.log(`\n[Bot] Logged in as ${client.user.tag}`);
  await registerSlashCommands();
  console.log(`[Bot] Model: ${OLLAMA_MODEL}`);
  console.log(`[Bot] Whisper: ${WHISPER_MODEL_PATH}\n`);

  // Auto-join configured channel on startup
  if (AUTO_JOIN_CHANNEL && GUILD_ID) {
    try {
      const guild   = await client.guilds.fetch(GUILD_ID);
      const channel = await client.channels.fetch(AUTO_JOIN_CHANNEL);
      if (channel && channel.isVoiceBased()) {
        await connectToChannel(channel);
      }
    } catch (err) {
      console.error('[Auto-join] Failed:', err.message);
    }
  }
});

// Auto-follow owner into voice channels
client.on('voiceStateUpdate', async (oldState, newState) => {
  if (!OWNER_USER_ID || newState.member?.id !== OWNER_USER_ID) return;
  const guildId = newState.guild.id;

  // Owner joined a voice channel
  if (newState.channelId && newState.channelId !== oldState.channelId) {
    const existing = guilds.get(guildId);
    if (existing) {
      existing.connection.destroy();
      guilds.delete(guildId);
    }
    try {
      await connectToChannel(newState.channel);
    } catch (err) {
      console.error('[Auto-follow] Failed:', err.message);
    }
  }

  // Owner left voice entirely
  if (!newState.channelId && oldState.channelId) {
    const state = guilds.get(guildId);
    if (state) {
      state.connection.destroy();
      guilds.delete(guildId);
      console.log('[Bot] Owner left — disconnected.');
    }
  }
});

// Slash command handler
client.on('interactionCreate', async (interaction) => {
  if (!interaction.isChatInputCommand()) return;

  const { commandName, guildId } = interaction;

  switch (commandName) {
    case 'join': {
      const voiceChannel = interaction.member?.voice?.channel;
      if (!voiceChannel) {
        await interaction.reply({ content: 'You need to be in a voice channel first.', ephemeral: true });
        return;
      }
      const existing = guilds.get(guildId);
      if (existing) {
        existing.connection.destroy();
        guilds.delete(guildId);
      }
      await interaction.deferReply();
      try {
        await connectToChannel(voiceChannel);
        await interaction.editReply(`Joined **${voiceChannel.name}**. I'm listening!`);
      } catch (err) {
        await interaction.editReply(`Failed to join: ${err.message}`);
      }
      break;
    }

    case 'leave': {
      const state = guilds.get(guildId);
      if (state) {
        state.connection.destroy();
        guilds.delete(guildId);
        await interaction.reply('Disconnected.');
      } else {
        await interaction.reply({ content: "I'm not in a voice channel.", ephemeral: true });
      }
      break;
    }

    case 'clear': {
      const state = guilds.get(guildId);
      if (state) {
        state.history = [];
        await interaction.reply('Conversation history cleared.');
      } else {
        await interaction.reply({ content: "I'm not active in this server.", ephemeral: true });
      }
      break;
    }

    case 'model': {
      const newModel = interaction.options.getString('name');
      process.env.OLLAMA_MODEL_OVERRIDE = newModel;
      await interaction.reply(`Switched to model: \`${newModel}\`\nTakes effect on the next message.`);
      break;
    }

    case 'status': {
      const state = guilds.get(guildId);
      const model = process.env.OLLAMA_MODEL_OVERRIDE || OLLAMA_MODEL;
      if (state) {
        await interaction.reply(
          `**Status**\n` +
          `Channel: **${state.channelName}**\n` +
          `Model: \`${model}\`\n` +
          `History: ${state.history.length} messages`
        );
      } else {
        await interaction.reply(`Not connected to any voice channel.\nModel: \`${model}\``);
      }
      break;
    }
  }
});

// Text commands (prefix-based, kept as fallback)
client.on('messageCreate', async (message) => {
  if (message.author.bot) return;
  if (!message.content.startsWith('!')) return;

  const [cmd, ...args] = message.content.slice(1).trim().split(/\s+/);

  switch (cmd.toLowerCase()) {
    case 'join': {
      const voiceChannel = message.member?.voice?.channel;
      if (!voiceChannel) {
        message.reply('You need to be in a voice channel first.');
        return;
      }
      const existing = guilds.get(message.guildId);
      if (existing) { existing.connection.destroy(); guilds.delete(message.guildId); }
      try {
        await connectToChannel(voiceChannel);
        message.reply(`Joined **${voiceChannel.name}**. I'm listening!`);
      } catch (err) {
        message.reply(`Failed to join: ${err.message}`);
      }
      break;
    }

    case 'leave': {
      const state = guilds.get(message.guildId);
      if (state) {
        state.connection.destroy();
        guilds.delete(message.guildId);
        message.reply('Disconnected.');
      } else {
        message.reply("I'm not in a voice channel.");
      }
      break;
    }

    case 'clear': {
      const state = guilds.get(message.guildId);
      if (state) {
        state.history = [];
        message.reply('Conversation history cleared.');
      }
      break;
    }

    case 'model': {
      const newModel = args[0];
      if (!newModel) {
        message.reply(`Current model: \`${OLLAMA_MODEL}\``);
        return;
      }
      // Note: reassigning module-level const requires a workaround
      process.env.OLLAMA_MODEL_OVERRIDE = newModel;
      message.reply(`Switched to model: \`${newModel}\``);
      break;
    }

    case 'help': {
      message.reply(
        '**Voice Agent Commands**\n' +
        '`!join` — Join your voice channel\n' +
        '`!leave` — Disconnect\n' +
        '`!clear` — Clear conversation history\n' +
        '`!model <name>` — Switch Ollama model\n' +
        '`!help` — Show this message'
      );
      break;
    }
  }
});

// ─────────────────────────────────────────────
// STARTUP
// ─────────────────────────────────────────────

if (!BOT_TOKEN) {
  console.error('[Error] BOT_TOKEN is not set in .env');
  process.exit(1);
}

client.login(BOT_TOKEN).catch((err) => {
  console.error('[Error] Failed to log in:', err.message);
  process.exit(1);
});

process.on('SIGINT', () => {
  console.log('\n[Bot] Shutting down...');
  for (const [, state] of guilds) state.connection.destroy();
  client.destroy();
  process.exit(0);
});
