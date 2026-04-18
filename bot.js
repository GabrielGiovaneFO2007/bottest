/**
 * Discord Voice Agent Bot
 * Listens to voice channels, transcribes with whisper.cpp,
 * queries Ollama, and speaks responses with Kokoro TTS.
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
const prism  = require('prism-media');
const fs     = require('fs');
const path   = require('path');
const os     = require('os');
const { execFile } = require('child_process');
const axios  = require('axios');

// ─────────────────────────────────────────────
// CONFIG  (all values come from .env)
// ─────────────────────────────────────────────

const BOT_TOKEN          = process.env.BOT_TOKEN;
const GUILD_ID           = process.env.GUILD_ID;
const AUTO_JOIN_CHANNEL  = process.env.AUTO_JOIN_CHANNEL_ID || null;
const OWNER_USER_ID      = process.env.OWNER_USER_ID        || null;
const OLLAMA_URL         = process.env.OLLAMA_URL   || 'http://localhost:11434/api/chat';
const OLLAMA_MODEL       = process.env.OLLAMA_MODEL || 'llama3.2:3b';
const WHISPER_BIN        = process.env.WHISPER_BIN  || path.join(os.homedir(), 'whisper.cpp/build/bin/whisper-cli');
const WHISPER_MODEL_PATH = process.env.WHISPER_MODEL || path.join(os.homedir(), 'whisper.cpp/models/ggml-base.bin');
const WHISPER_LANG       = process.env.WHISPER_LANG || 'pt';
const KOKORO_MODEL       = process.env.KOKORO_MODEL || path.join(os.homedir(), 'kokoro-v1.0.onnx');
const KOKORO_VOICES      = process.env.KOKORO_VOICES || path.join(os.homedir(), 'voices-v1.0.bin');

const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT ||
  'Você é um assistente de voz útil dentro de um canal de voz do Discord. ' +
  'Suas respostas são faladas em voz alta, então seja conversacional e conciso. ' +
  'Evite markdown, listas e blocos de código a menos que o usuário peça. ' +
  'Chame a pessoa pelo nome de usuário do Discord ao responder.';

// ── Audio constants ───────────────────────────
// Discord sends 48kHz stereo Opus; Whisper needs 16kHz mono PCM
const DISCORD_SAMPLE_RATE   = 48000;
const DISCORD_CHANNELS      = 2;
const WHISPER_SAMPLE_RATE   = 16000;
const FRAME_SIZE            = 960;

// ── Sensitivity tuning ────────────────────────
const SILENCE_DURATION_MS   = 2500;   // ms of silence before treating a clip as finished
const MIN_AUDIO_DURATION_MS = 1500;   // discard clips shorter than this (coughs, noise, etc.)
const DEBOUNCE_MS           = 600;    // wait after clip ends before processing
                                       // if speaker resumes within this window, clips merge

// ─────────────────────────────────────────────
// SLASH COMMANDS
// ─────────────────────────────────────────────

const commands = [
  new SlashCommandBuilder()
    .setName('join')
    .setDescription('Bot entra no seu canal de voz'),

  new SlashCommandBuilder()
    .setName('leave')
    .setDescription('Bot sai do canal de voz'),

  new SlashCommandBuilder()
    .setName('clear')
    .setDescription('Limpa o histórico de conversa'),

  new SlashCommandBuilder()
    .setName('model')
    .setDescription('Troca o modelo do Ollama')
    .addStringOption(opt =>
      opt.setName('name')
        .setDescription('Nome do modelo (ex: llama3.2:3b, qwen2.5:3b)')
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName('status')
    .setDescription('Mostra o status atual do bot'),

  new SlashCommandBuilder()
    .setName('sensitivity')
    .setDescription('Ajusta a sensibilidade do microfone em tempo real')
    .addNumberOption(opt =>
      opt.setName('value')
        .setDescription('Threshold de silêncio (padrão 0.04 — maior = menos sensível)')
        .setMinValue(0.01)
        .setMaxValue(0.2)
        .setRequired(true)
    ),
].map(cmd => cmd.toJSON());

async function registerSlashCommands() {
  const rest = new REST({ version: '10' }).setToken(BOT_TOKEN);
  try {
    console.log('[Slash] Registrando comandos...');
    await rest.put(
      Routes.applicationGuildCommands(process.env.CLIENT_ID, GUILD_ID),
      { body: commands }
    );
    console.log('[Slash] Comandos registrados.');
  } catch (err) {
    console.error('[Slash] Falha ao registrar comandos:', err.message);
  }
}

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

// Per-guild: { connection, player, history, channelName }
const guilds = new Map();

// Live-adjustable silence threshold — change with /sensitivity or !sensitivity
let currentThreshold = 0.04;

// ─────────────────────────────────────────────
// DISCORD CLIENT
// ─────────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
  ],
});

// ─────────────────────────────────────────────
// AUDIO UTILITIES
// ─────────────────────────────────────────────

/**
 * Downsample 48kHz stereo Int16 PCM → 16kHz mono Int16 PCM
 */
function downsample(buffer) {
  const input  = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  const ratio  = DISCORD_SAMPLE_RATE / WHISPER_SAMPLE_RATE;
  const outLen = Math.floor(input.length / DISCORD_CHANNELS / ratio);
  const output = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcIndex = Math.floor(i * ratio) * DISCORD_CHANNELS;
    output[i] = (input[srcIndex] + input[srcIndex + 1]) >> 1;
  }

  return Buffer.from(output.buffer);
}

/**
 * Calculate RMS (volume level) of a PCM buffer.
 * Used to filter out silent/noise-only clips before sending to Whisper.
 */
function calcRms(buffer) {
  const samples = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i] / 32768;
    sum += s * s;
  }
  return Math.sqrt(sum / samples.length);
}

/**
 * Write a raw PCM buffer to a WAV file (16kHz mono Int16).
 */
function writePcmToWav(pcmBuffer, filePath) {
  return new Promise((resolve, reject) => {
    const dataSize      = pcmBuffer.length;
    const sampleRate    = WHISPER_SAMPLE_RATE;
    const numChannels   = 1;
    const bitsPerSample = 16;
    const byteRate      = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign    = numChannels * bitsPerSample / 8;
    const header        = Buffer.alloc(44);

    header.write('RIFF', 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
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
      [
        '-m', WHISPER_MODEL_PATH,
        '-f', wavPath,
        '--language', WHISPER_LANG,
        '--no-timestamps',
        '-nt',
      ],
      { timeout: 30000 },
      (err, stdout) => {
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

  const model = process.env.OLLAMA_MODEL_OVERRIDE || OLLAMA_MODEL;

  state.history.push({
    role: 'user',
    content: `[${username}]: ${userText}`,
  });

  try {
    const response = await axios.post(
      OLLAMA_URL,
      {
        model,
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

    // Keep last 20 messages to avoid context overflow
    if (state.history.length > 20) {
      state.history = state.history.slice(-20);
    }

    return reply;
  } catch (err) {
    console.error('[Ollama] Erro:', err.message);
    return 'Desculpe, não consegui acessar o modelo de IA agora.';
  }
}

// ─────────────────────────────────────────────
// TTS (Kokoro — bilingual PT-BR + EN)
// ─────────────────────────────────────────────

function generateTts(text, outputPath) {
  return new Promise((resolve, reject) => {
    // Simple language detection: if text is mostly ASCII letters → English
    const asciiRatio = (text.match(/[a-zA-Z]/g) || []).length / Math.max(text.length, 1);
    const lang  = asciiRatio > 0.85 ? 'en-us' : 'pt-br';
    const voice = lang === 'en-us' ? 'bf_emma' : 'pf_dora';

    const safeText = text
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
      .replace(/\n/g, ' ');

    const script = `
from kokoro_onnx import Kokoro
import soundfile as sf
k = Kokoro("${KOKORO_MODEL}", "${KOKORO_VOICES}")
samples, sr = k.create("${safeText}", voice="${voice}", speed=1.0, lang="${lang}")
sf.write("${outputPath}", samples, sr)
`;

    execFile('python3', ['-c', script], { timeout: 30000 }, (err) => {
      if (err) reject(err); else resolve();
    });
  });
}

// ─────────────────────────────────────────────
// AUDIO PLAYER
// ─────────────────────────────────────────────

function playAudio(guildId, audioPath) {
  const state = guilds.get(guildId);
  if (!state?.player) return Promise.resolve();

  return new Promise((resolve) => {
    const resource = createAudioResource(audioPath, { inputType: StreamType.Arbitrary });
    state.player.play(resource);
    state.player.once(AudioPlayerStatus.Idle, resolve);
    state.player.once('error', (err) => {
      console.error('[Player] Erro:', err.message);
      resolve();
    });
  });
}

// ─────────────────────────────────────────────
// PIPELINE — filter → transcribe → LLM → TTS → play
// ─────────────────────────────────────────────

async function processAudio(guildId, userId, username, pcmChunks) {
  const tmpDir  = os.tmpdir();
  const wavPath = path.join(tmpDir, `discord_${userId}_${Date.now()}.wav`);
  const ttsPath = path.join(tmpDir, `tts_${Date.now()}.wav`);

  try {
    const rawPcm   = Buffer.concat(pcmChunks);
    const pcm16khz = downsample(rawPcm);

    // 1. Duration filter
    const durationMs = (pcm16khz.length / 2 / WHISPER_SAMPLE_RATE) * 1000;
    if (durationMs < MIN_AUDIO_DURATION_MS) {
      console.log(`[Filter] Ignorado: ${Math.round(durationMs)}ms (abaixo do mínimo)`);
      return;
    }

    // 2. Volume filter — discard clips that are mostly silence
    const rms = calcRms(pcm16khz);
    if (rms < currentThreshold) {
      console.log(`[Filter] Ignorado: RMS ${rms.toFixed(4)} abaixo do threshold ${currentThreshold}`);
      return;
    }

    // 3. Write WAV and transcribe
    await writePcmToWav(pcm16khz, wavPath);
    console.log(`[Whisper] Transcrevendo ${username} (${Math.round(durationMs)}ms, RMS ${rms.toFixed(4)})...`);

    const text = await transcribe(wavPath);
    if (!text || text.length < 3) {
      console.log('[Whisper] Transcrição vazia, ignorando.');
      return;
    }

    console.log(`[${username}]: ${text}`);

    // 4. Query LLM
    console.log('[Ollama] Consultando...');
    const reply = await queryOllama(guildId, username, text);
    if (!reply) return;
    console.log(`[Bot]: ${reply}\n`);

    // 5. TTS and play
    await generateTts(reply, ttsPath);
    await playAudio(guildId, ttsPath);

  } catch (err) {
    console.error('[Pipeline] Erro:', err.message);
  } finally {
    for (const f of [wavPath, ttsPath]) fs.unlink(f, () => {});
  }
}

// ─────────────────────────────────────────────
// VOICE CONNECTION
// ─────────────────────────────────────────────

async function connectToChannel(channel) {
  const guildId = channel.guild.id;

  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
  });

  const player = createAudioPlayer();
  connection.subscribe(player);

  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
  } catch {
    connection.destroy();
    throw new Error('Não foi possível conectar ao canal em 20 segundos.');
  }

  guilds.set(guildId, {
    connection,
    player,
    history: [],
    channelName: channel.name,
  });

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
  console.log(`[Bot] Entrou em #${channel.name} em ${channel.guild.name}`);
}

// ─────────────────────────────────────────────
// AUDIO RECEIVER + DEBOUNCE LOGIC
// ─────────────────────────────────────────────

function startListening(guildId) {
  const state = guilds.get(guildId);
  if (!state) return;

  const { connection } = state;
  const receiver = connection.receiver;

  // Per-user debounce timers and accumulated audio chunks
  const pendingTimers = new Map();
  const pendingChunks = new Map();

  // Single processing queue per guild — responses play one at a time
  const queue    = [];
  let processing = false;

  function drainQueue() {
    if (processing || queue.length === 0) return;
    processing = true;
    const job = queue.shift();
    processAudio(job.guildId, job.userId, job.username, job.chunks).finally(() => {
      processing = false;
      drainQueue();
    });
  }

  receiver.speaking.on('start', (userId) => {
    if (userId === client.user.id) return;

    // Cancel the pending processing timer for this user — they're still talking
    if (pendingTimers.has(userId)) {
      clearTimeout(pendingTimers.get(userId));
      pendingTimers.delete(userId);
    }

    const member   = client.guilds.cache.get(guildId)?.members.cache.get(userId);
    const username = member?.displayName || member?.user?.username || `User ${userId}`;

    const audioStream = receiver.subscribe(userId, {
      end: {
        behavior: EndBehaviorType.AfterSilence,
        duration: SILENCE_DURATION_MS,
      },
    });

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

        // Merge with any chunks already waiting for this user
        // (handles brief mid-sentence pauses)
        const existing = pendingChunks.get(userId) || [];
        const merged   = [...existing, ...chunks];
        pendingChunks.set(userId, merged);

        // Wait DEBOUNCE_MS before processing. If they speak again before
        // the timer fires, it gets cancelled and chunks keep accumulating.
        const timer = setTimeout(() => {
          pendingTimers.delete(userId);
          const finalChunks = pendingChunks.get(userId) || [];
          pendingChunks.delete(userId);
          if (finalChunks.length === 0) return;

          if (!processing) {
            processing = true;
            processAudio(guildId, userId, username, finalChunks).finally(() => {
              processing = false;
              drainQueue();
            });
          } else {
            queue.push({ guildId, userId, username, chunks: finalChunks });
          }
        }, DEBOUNCE_MS);

        pendingTimers.set(userId, timer);
      })
      .on('error', (err) => console.error('[Decoder] Erro:', err.message));
  });
}

// ─────────────────────────────────────────────
// DISCORD EVENTS
// ─────────────────────────────────────────────

client.once('ready', async () => {
  console.log(`\n[Bot] Logado como ${client.user.tag}`);
  await registerSlashCommands();
  console.log(`[Bot] Modelo: ${OLLAMA_MODEL}`);
  console.log(`[Bot] Whisper: ${WHISPER_MODEL_PATH}`);
  console.log(`[Bot] Idioma: ${WHISPER_LANG}`);
  console.log(`[Bot] Threshold inicial: ${currentThreshold}\n`);

  if (AUTO_JOIN_CHANNEL && GUILD_ID) {
    try {
      await client.guilds.fetch(GUILD_ID);
      const channel = await client.channels.fetch(AUTO_JOIN_CHANNEL);
      if (channel?.isVoiceBased()) await connectToChannel(channel);
    } catch (err) {
      console.error('[Auto-join] Falhou:', err.message);
    }
  }
});

// Auto-follow owner into voice channels
client.on('voiceStateUpdate', async (oldState, newState) => {
  if (!OWNER_USER_ID || newState.member?.id !== OWNER_USER_ID) return;
  const guildId = newState.guild.id;

  if (newState.channelId && newState.channelId !== oldState.channelId) {
    const existing = guilds.get(guildId);
    if (existing) { existing.connection.destroy(); guilds.delete(guildId); }
    try {
      await connectToChannel(newState.channel);
    } catch (err) {
      console.error('[Auto-follow] Falhou:', err.message);
    }
  }

  if (!newState.channelId && oldState.channelId) {
    const state = guilds.get(guildId);
    if (state) {
      state.connection.destroy();
      guilds.delete(guildId);
      console.log('[Bot] Dono saiu — desconectado.');
    }
  }
});

// ─────────────────────────────────────────────
// SLASH COMMAND HANDLER
// ─────────────────────────────────────────────

client.on('interactionCreate', async (interaction) => {
  if (!interaction.isChatInputCommand()) return;

  const { commandName, guildId } = interaction;

  switch (commandName) {
    case 'join': {
      const voiceChannel = interaction.member?.voice?.channel;
      if (!voiceChannel) {
        await interaction.reply({ content: 'Você precisa estar em um canal de voz primeiro.', ephemeral: true });
        return;
      }
      const existing = guilds.get(guildId);
      if (existing) { existing.connection.destroy(); guilds.delete(guildId); }
      await interaction.deferReply();
      try {
        await connectToChannel(voiceChannel);
        await interaction.editReply(`Entrei em **${voiceChannel.name}**. Estou ouvindo!`);
      } catch (err) {
        await interaction.editReply(`Falha ao entrar: ${err.message}`);
      }
      break;
    }

    case 'leave': {
      const state = guilds.get(guildId);
      if (state) {
        state.connection.destroy();
        guilds.delete(guildId);
        await interaction.reply('Desconectado.');
      } else {
        await interaction.reply({ content: 'Não estou em nenhum canal de voz.', ephemeral: true });
      }
      break;
    }

    case 'clear': {
      const state = guilds.get(guildId);
      if (state) {
        state.history = [];
        await interaction.reply('Histórico de conversa limpo.');
      } else {
        await interaction.reply({ content: 'Não estou ativo neste servidor.', ephemeral: true });
      }
      break;
    }

    case 'model': {
      const newModel = interaction.options.getString('name');
      process.env.OLLAMA_MODEL_OVERRIDE = newModel;
      await interaction.reply(`Modelo trocado para: \`${newModel}\`\nVai usar na próxima mensagem.`);
      break;
    }

    case 'status': {
      const state = guilds.get(guildId);
      const model = process.env.OLLAMA_MODEL_OVERRIDE || OLLAMA_MODEL;
      if (state) {
        await interaction.reply(
          `**Status**\n` +
          `Canal: **${state.channelName}**\n` +
          `Modelo: \`${model}\`\n` +
          `Idioma: \`${WHISPER_LANG}\`\n` +
          `Threshold: \`${currentThreshold}\`\n` +
          `Histórico: ${state.history.length} mensagens`
        );
      } else {
        await interaction.reply(`Não conectado a nenhum canal.\nModelo: \`${model}\``);
      }
      break;
    }

    case 'sensitivity': {
      const value = interaction.options.getNumber('value');
      currentThreshold = value;
      await interaction.reply(
        `Sensibilidade ajustada para \`${value}\`.\n` +
        `(menor = mais sensível | maior = ignora mais ruído de fundo)`
      );
      break;
    }
  }
});

// ─────────────────────────────────────────────
// STARTUP
// ─────────────────────────────────────────────

if (!BOT_TOKEN) {
  console.error('[Erro] BOT_TOKEN não definido no .env');
  process.exit(1);
}

client.login(BOT_TOKEN).catch((err) => {
  console.error('[Erro] Falha no login:', err.message);
  process.exit(1);
});

process.on('SIGINT', () => {
  console.log('\n[Bot] Encerrando...');
  for (const [, state] of guilds) state.connection.destroy();
  client.destroy();
  process.exit(0);
});
