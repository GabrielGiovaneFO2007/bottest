/**
 * Discord Voice Agent Bot
 *
 * Required .env keys:
 * BOT_TOKEN
 * CLIENT_ID
 * GUILD_ID
 * AUTO_JOIN_CHANNEL_ID
 * OWNER_USER_ID
 * OLLAMA_URL
 * OLLAMA_MODEL
 * WHISPER_BIN
 * WHISPER_MODEL
 * WHISPER_LANG
 * TTS_ENGINE=piper|server
 * PIPER_BIN
 * PIPER_MODEL
 * PIPER_LENGTH_SCALE
 * PIPER_NOISE_SCALE
 * PIPER_NOISE_W
 * TTS_SERVER_URL
 * RMS_THRESHOLD
 * MIN_AUDIO_MS
 * SILENCE_MS
 * DEBOUNCE_MS
 * SYSTEM_PROMPT
 * PREFIX
 *
 * Notes:
 * - Use a Discord.js / @discordjs/voice version compatible with your Node version.
 * - If you run Node 20, do not keep an @discordjs/voice release that requires Node 22+.
 * - This file supports both slash commands and prefix commands.
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
const fs = require('fs');
const path = require('path');
const os = require('os');
const axios = require('axios');
const { execFile, spawn } = require('child_process');

// ─────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────

const CFG = {
  BOT_TOKEN: process.env.BOT_TOKEN || '',
  CLIENT_ID: process.env.CLIENT_ID || '',
  GUILD_ID: process.env.GUILD_ID || '',

  AUTO_JOIN_CHANNEL_ID: process.env.AUTO_JOIN_CHANNEL_ID || '',
  OWNER_USER_ID: process.env.OWNER_USER_ID || '',

  OLLAMA_URL: process.env.OLLAMA_URL || 'http://localhost:11434/api/chat',
  OLLAMA_MODEL: process.env.OLLAMA_MODEL || 'llama3.2:3b',
  OLLAMA_MAX_HISTORY: Number(process.env.OLLAMA_MAX_HISTORY || '20'),

  WHISPER_BIN: process.env.WHISPER_BIN || findFirstExisting([
    path.join(os.homedir(), 'whisper.cpp/build/bin/whisper-cli'),
    path.join(os.homedir(), 'whisper.cpp/build/bin/main'),
    path.join(os.homedir(), 'whisper.cpp/main'),
  ]),
  WHISPER_MODEL: process.env.WHISPER_MODEL || path.join(os.homedir(), 'whisper.cpp/models/ggml-base.en.bin'),
  WHISPER_LANG: process.env.WHISPER_LANG || 'pt',

  TTS_ENGINE: (process.env.TTS_ENGINE || 'piper').toLowerCase(),
  TTS_SERVER_URL: process.env.TTS_SERVER_URL || '',

  PIPER_BIN: process.env.PIPER_BIN || path.join(os.homedir(), 'piper/piper/piper'),
  PIPER_MODEL: process.env.PIPER_MODEL || path.join(os.homedir(), 'piper/voices/pt_BR-faber-medium.onnx'),
  PIPER_LENGTH_SCALE: process.env.PIPER_LENGTH_SCALE || '1.15',
  PIPER_NOISE_SCALE: process.env.PIPER_NOISE_SCALE || '0.6',
  PIPER_NOISE_W: process.env.PIPER_NOISE_W || '0.8',

  RMS_THRESHOLD: Number(process.env.RMS_THRESHOLD || '0.015'),
  MIN_AUDIO_MS: Number(process.env.MIN_AUDIO_MS || '600'),
  SILENCE_MS: Number(process.env.SILENCE_MS || '1000'),
  DEBOUNCE_MS: Number(process.env.DEBOUNCE_MS || '300'),

  SYSTEM_PROMPT: process.env.SYSTEM_PROMPT ||
    'Você é um assistente de voz útil dentro de um canal do Discord. Responda de forma natural, clara e curta. ' +
    'Se o usuário falar em português, responda em português. Evite markdown quando não for necessário.',

  PREFIX: process.env.PREFIX || '!',
  DEBUG: String(process.env.DEBUG || '').toLowerCase() === '1',
};

const DISCORD_SAMPLE_RATE = 48000;
const DISCORD_CHANNELS = 2;
const WHISPER_SAMPLE_RATE = 16000;
const FRAME_SIZE = 960;

if (!CFG.BOT_TOKEN) {
  console.error('[Config] BOT_TOKEN is missing');
  process.exit(1);
}
if (!CFG.CLIENT_ID) {
  console.error('[Config] CLIENT_ID is missing');
  process.exit(1);
}
if (!CFG.GUILD_ID) {
  console.error('[Config] GUILD_ID is missing');
  process.exit(1);
}

// ─────────────────────────────────────────────────────────────
// DISCORD CLIENT
// ─────────────────────────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// Per guild state
const guilds = new Map();

// ─────────────────────────────────────────────────────────────
// SLASH COMMANDS
// ─────────────────────────────────────────────────────────────

const slashCommands = [
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
    .addStringOption((opt) =>
      opt.setName('name')
        .setDescription('Ex: llama3.2:3b, qwen2.5:3b')
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName('status')
    .setDescription('Mostra o status atual do bot'),

  new SlashCommandBuilder()
    .setName('sensitivity')
    .setDescription('Ajusta o limiar RMS')
    .addNumberOption((opt) =>
      opt.setName('value')
        .setDescription('Menor = mais sensível')
        .setMinValue(0.001)
        .setMaxValue(0.2)
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName('silence')
    .setDescription('Ajusta o tempo de silêncio antes de processar')
    .addIntegerOption((opt) =>
      opt.setName('value')
        .setDescription('Milissegundos')
        .setMinValue(200)
        .setMaxValue(5000)
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName('debug')
    .setDescription('Mostra a configuração atual do bot'),
].map((cmd) => cmd.toJSON());

async function registerSlashCommands() {
  const rest = new REST({ version: '10' }).setToken(CFG.BOT_TOKEN);
  await rest.put(
    Routes.applicationGuildCommands(CFG.CLIENT_ID, CFG.GUILD_ID),
    { body: slashCommands }
  );
}

// ─────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────

function log(...args) {
  console.log(...args);
}

function findFirstExisting(candidates) {
  for (const candidate of candidates) {
    try {
      if (candidate && fs.existsSync(candidate)) return candidate;
    } catch {
      // ignore
    }
  }
  return candidates[0];
}

function safeDelete(filePath) {
  fs.unlink(filePath, () => {});
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeText(text) {
  return String(text || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`/g, ' ')
    .replace(/\*\*/g, ' ')
    .replace(/\*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitCompleteSentences(buffer, final = false) {
  const sentences = [];
  let rest = buffer.trimStart();

  while (rest.length > 0) {
    const match = rest.match(/^([\s\S]*?[.!?]+)(?:\s|$)/);
    if (!match) break;
    const sentence = match[1].trim();
    if (sentence) sentences.push(sentence);
    rest = rest.slice(match[0].length).trimStart();
  }

  if (final && rest.trim()) {
    sentences.push(rest.trim());
    rest = '';
  }

  return { sentences, rest };
}

function ensureGuildState(guildId) {
  let state = guilds.get(guildId);
  if (!state) {
    state = {
      connection: null,
      player: null,
      history: [],
      channelName: '',
      ttsBusy: false,
      speechQueue: [],
    };
    guilds.set(guildId, state);
  }
  return state;
}

// ─────────────────────────────────────────────────────────────
// AUDIO UTILITIES
// ─────────────────────────────────────────────────────────────

function downsampleDiscordPcmToWhisper(buffer) {
  const input = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  const ratio = DISCORD_SAMPLE_RATE / WHISPER_SAMPLE_RATE;
  const outLen = Math.floor(input.length / DISCORD_CHANNELS / ratio);
  const output = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcIndex = Math.floor(i * ratio) * DISCORD_CHANNELS;
    const left = input[srcIndex] || 0;
    const right = input[srcIndex + 1] || 0;
    output[i] = (left + right) >> 1;
  }

  return Buffer.from(output.buffer);
}

function calcRms(buffer) {
  const samples = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  if (!samples.length) return 0;

  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i] / 32768;
    sum += s * s;
  }
  return Math.sqrt(sum / samples.length);
}

function writePcmToWav(pcmBuffer, filePath) {
  return new Promise((resolve, reject) => {
    const dataSize = pcmBuffer.length;
    const sampleRate = WHISPER_SAMPLE_RATE;
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
    const blockAlign = (numChannels * bitsPerSample) / 8;
    const header = Buffer.alloc(44);

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
      if (err) reject(err);
      else resolve();
    });
  });
}

// ─────────────────────────────────────────────────────────────
// WHISPER
// ─────────────────────────────────────────────────────────────

function transcribe(wavPath) {
  return new Promise((resolve, reject) => {
    if (!CFG.WHISPER_BIN || !fs.existsSync(CFG.WHISPER_BIN)) {
      reject(new Error(`Whisper binary not found: ${CFG.WHISPER_BIN}`));
      return;
    }
    if (!fs.existsSync(CFG.WHISPER_MODEL)) {
      reject(new Error(`Whisper model not found: ${CFG.WHISPER_MODEL}`));
      return;
    }

    execFile(
      CFG.WHISPER_BIN,
      [
        '-m', CFG.WHISPER_MODEL,
        '-f', wavPath,
        '--language', CFG.WHISPER_LANG,
        '--no-timestamps',
        '-nt',
      ],
      { timeout: 30000 },
      (err, stdout, stderr) => {
        if (err) {
          reject(new Error(stderr || err.message));
          return;
        }

        const text = stdout
          .split('\n')
          .map((line) => line.replace(/\[.*?\]/g, '').trim())
          .filter(Boolean)
          .join(' ')
          .trim();

        resolve(normalizeText(text));
      }
    );
  });
}

// ─────────────────────────────────────────────────────────────
// OLLAMA
// ─────────────────────────────────────────────────────────────

async function queryOllamaNonStreaming(guildId, username, userText) {
  const state = ensureGuildState(guildId);
  const model = process.env.OLLAMA_MODEL_OVERRIDE || CFG.OLLAMA_MODEL;

  state.history.push({
    role: 'user',
    content: `[${username}]: ${userText}`,
  });

  try {
    const response = await axios.post(
      CFG.OLLAMA_URL,
      {
        model,
        messages: [
          { role: 'system', content: CFG.SYSTEM_PROMPT },
          ...state.history,
        ],
        stream: false,
      },
      { timeout: 60000 }
    );

    const reply = normalizeText(response.data?.message?.content || '');
    if (!reply) return null;

    state.history.push({ role: 'assistant', content: reply });
    if (state.history.length > CFG.OLLAMA_MAX_HISTORY) {
      state.history = state.history.slice(-CFG.OLLAMA_MAX_HISTORY);
    }

    return reply;
  } catch (err) {
    console.error('[Ollama] Non-stream error:', err.message);
    return 'Desculpe, não consegui acessar o modelo de IA agora.';
  }
}

async function queryOllamaStream(guildId, username, userText, onSentence) {
  const state = ensureGuildState(guildId);
  const model = process.env.OLLAMA_MODEL_OVERRIDE || CFG.OLLAMA_MODEL;

  state.history.push({
    role: 'user',
    content: `[${username}]: ${userText}`,
  });

  let response;
  try {
    response = await axios.post(
      CFG.OLLAMA_URL,
      {
        model,
        messages: [
          { role: 'system', content: CFG.SYSTEM_PROMPT },
          ...state.history,
        ],
        stream: true,
      },
      { responseType: 'stream', timeout: 60000 }
    );
  } catch (err) {
    console.error('[Ollama] Streaming request failed:', err.message);
    const fallback = await queryOllamaNonStreaming(guildId, username, userText);
    if (fallback) onSentence(fallback, true);
    return fallback || '';
  }

  let ndjsonBuffer = '';
  let assistantText = '';
  let sentenceBuffer = '';

  const flushSentences = (final = false) => {
    const extracted = splitCompleteSentences(sentenceBuffer, final);
    sentenceBuffer = extracted.rest;

    for (const sentence of extracted.sentences) {
      const clean = normalizeText(sentence);
      if (clean) {
        assistantText += (assistantText ? ' ' : '') + clean;
        onSentence(clean, false);
      }
    }
  };

  await new Promise((resolve, reject) => {
    response.data.on('data', (chunk) => {
      ndjsonBuffer += chunk.toString('utf8');

      let newlineIndex;
      while ((newlineIndex = ndjsonBuffer.indexOf('\n')) !== -1) {
        const rawLine = ndjsonBuffer.slice(0, newlineIndex).trim();
        ndjsonBuffer = ndjsonBuffer.slice(newlineIndex + 1);
        if (!rawLine) continue;

        let obj;
        try {
          obj = JSON.parse(rawLine);
        } catch {
          continue;
        }

        const token = obj?.message?.content || '';
        if (token) {
          sentenceBuffer += token;
        }

        if (obj?.done) {
          flushSentences(true);
        } else {
          flushSentences(false);
        }
      }
    });

    response.data.on('end', () => {
      flushSentences(true);
      resolve();
    });

    response.data.on('error', reject);
  });

  const finalReply = normalizeText(assistantText || sentenceBuffer);
  if (finalReply) {
    state.history.push({ role: 'assistant', content: finalReply });
    if (state.history.length > CFG.OLLAMA_MAX_HISTORY) {
      state.history = state.history.slice(-CFG.OLLAMA_MAX_HISTORY);
    }
  }

  return finalReply;
}

// ─────────────────────────────────────────────────────────────
// TTS
// ─────────────────────────────────────────────────────────────

async function generateTtsViaServer(text, outputPath) {
  const asciiRatio = (text.match(/[a-zA-Z]/g) || []).length / Math.max(text.length, 1);
  const lang = asciiRatio > 0.85 ? 'en-us' : 'pt-br';

  const response = await axios.post(
    CFG.TTS_SERVER_URL,
    { text, lang },
    { responseType: 'arraybuffer', timeout: 20000 }
  );

  fs.writeFileSync(outputPath, Buffer.from(response.data));
}

function generateTtsLocal(text, outputPath) {
  return new Promise((resolve, reject) => {
    if (!CFG.PIPER_BIN || !fs.existsSync(CFG.PIPER_BIN)) {
      reject(new Error(`Piper binary not found: ${CFG.PIPER_BIN}`));
      return;
    }
    if (!fs.existsSync(CFG.PIPER_MODEL)) {
      reject(new Error(`Piper model not found: ${CFG.PIPER_MODEL}`));
      return;
    }

    const args = [
      '--model', CFG.PIPER_MODEL,
      '--output_file', outputPath,
      '--length_scale', String(CFG.PIPER_LENGTH_SCALE),
      '--noise_scale', String(CFG.PIPER_NOISE_SCALE),
      '--noise_w', String(CFG.PIPER_NOISE_W),
    ];

    const proc = spawn(CFG.PIPER_BIN, args, { stdio: ['pipe', 'pipe', 'pipe'] });

    let stderr = '';
    proc.stderr.on('data', (d) => {
      stderr += d.toString();
      if (CFG.DEBUG) process.stdout.write(d);
    });

    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Piper exited with code ${code}`));
        return;
      }
      resolve();
    });

    proc.stdin.write(String(text || '').trim());
    proc.stdin.end();
  });
}

async function generateTts(text, outputPath) {
  const clean = normalizeText(text);
  if (!clean) throw new Error('Empty TTS text');

  if (CFG.TTS_ENGINE === 'server' && CFG.TTS_SERVER_URL) {
    return generateTtsViaServer(clean, outputPath);
  }

  return generateTtsLocal(clean, outputPath);
}

// ─────────────────────────────────────────────────────────────
// PLAYBACK
// ─────────────────────────────────────────────────────────────

function playAudio(guildId, audioPath) {
  const state = guilds.get(guildId);
  if (!state?.player || !state?.connection) return Promise.resolve();

  return new Promise((resolve) => {
    const stream = fs.createReadStream(audioPath);
    const resource = createAudioResource(stream, {
      inputType: StreamType.Arbitrary,
      inlineVolume: false,
    });

    const done = () => {
      state.player.off('error', onError);
      resolve();
    };

    const onError = (err) => {
      console.error('[Player] Error:', err.message);
      done();
    };

    state.player.once(AudioPlayerStatus.Idle, done);
    state.player.once('error', onError);
    state.player.play(resource);
  });
}

// ─────────────────────────────────────────────────────────────
// PIPELINE
// ─────────────────────────────────────────────────────────────

async function processAudio(guildId, userId, username, pcmChunks) {
  const state = guilds.get(guildId);
  if (!state) return;

  const tmpDir = os.tmpdir();
  const wavPath = path.join(tmpDir, `discord_${guildId}_${userId}_${Date.now()}.wav`);

  const ttsQueue = [];
  let ttsBusy = false;
  let streamFinished = false;
  let resolveDrain;
  const drainPromise = new Promise((resolve) => {
    resolveDrain = resolve;
  });

  const maybeResolveDrain = () => {
    if (streamFinished && !ttsBusy && ttsQueue.length === 0) {
      resolveDrain();
    }
  };

  const pumpTtsQueue = async () => {
    if (ttsBusy) return;
    ttsBusy = true;

    while (ttsQueue.length > 0) {
      const segment = normalizeText(ttsQueue.shift());
      if (!segment) continue;

      const ttsPath = path.join(
        tmpDir,
        `tts_${guildId}_${Date.now()}_${Math.random().toString(16).slice(2)}.wav`
      );

      try {
        if (CFG.DEBUG) console.log('[TTS] Segment:', segment);
        await generateTts(segment, ttsPath);
        await playAudio(guildId, ttsPath);
      } catch (err) {
        console.error('[TTS] Error:', err.message);
      } finally {
        safeDelete(ttsPath);
      }
    }

    ttsBusy = false;
    maybeResolveDrain();
  };

  const enqueueSegment = (segment) => {
    const clean = normalizeText(segment);
    if (!clean) return;
    ttsQueue.push(clean);
    void pumpTtsQueue();
  };

  try {
    const rawPcm = Buffer.concat(pcmChunks);
    const pcm16khz = downsampleDiscordPcmToWhisper(rawPcm);

    const durationMs = (pcm16khz.length / 2 / WHISPER_SAMPLE_RATE) * 1000;
    const rms = calcRms(pcm16khz);

    if (CFG.DEBUG) {
      console.log(`[DEBUG] ${username} duration=${Math.round(durationMs)}ms rms=${rms.toFixed(4)}`);
    }

    if (durationMs < CFG.MIN_AUDIO_MS) {
      console.log(`[Filter] Ignored ${username}: ${Math.round(durationMs)}ms below minimum`);
      return;
    }

    if (rms < CFG.RMS_THRESHOLD) {
      console.log(`[Filter] Ignored ${username}: RMS ${rms.toFixed(4)} below threshold ${CFG.RMS_THRESHOLD}`);
      return;
    }

    await writePcmToWav(pcm16khz, wavPath);
    console.log(`[Whisper] Transcribing ${username}...`);

    const text = await transcribe(wavPath);
    if (!text || text.length < 2) {
      console.log('[Whisper] Empty transcription, ignoring.');
      return;
    }

    console.log(`[${username}] ${text}`);

    console.log('[Ollama] Streaming reply...');
    const reply = await queryOllamaStream(guildId, username, text, enqueueSegment);

    if (!reply) {
      enqueueSegment('Desculpe, não consegui gerar uma resposta agora.');
    }

    streamFinished = true;
    maybeResolveDrain();
    await drainPromise;
  } catch (err) {
    console.error('[Pipeline] Error:', err.message);
  } finally {
    safeDelete(wavPath);
  }
}

// ─────────────────────────────────────────────────────────────
// VOICE CONNECTION
// ─────────────────────────────────────────────────────────────

async function connectToChannel(channel) {
  const guildId = channel.guild.id;
  const state = ensureGuildState(guildId);

  if (state.connection) {
    try { state.connection.destroy(); } catch {}
  }

  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
  });

  const player = createAudioPlayer();
  connection.subscribe(player);

  state.connection = connection;
  state.player = player;
  state.channelName = channel.name;

  player.on('error', (err) => {
    console.error('[Voice Player] Error:', err.message);
  });

  connection.on('error', (err) => {
    console.error('[Voice Connection] Error:', err.message);
  });

  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
  } catch {
    try { connection.destroy(); } catch {}
    state.connection = null;
    state.player = null;
    throw new Error('Could not connect to voice channel within 20 seconds.');
  }

  startListening(guildId);
  console.log(`[Bot] Joined #${channel.name} in ${channel.guild.name}`);
}

function disconnectGuild(guildId) {
  const state = guilds.get(guildId);
  if (!state) return;

  try {
    if (state.connection) state.connection.destroy();
  } catch {}

  state.connection = null;
  state.player = null;
  state.speechQueue = [];
  state.ttsBusy = false;
}

// ─────────────────────────────────────────────────────────────
// LISTENING
// ─────────────────────────────────────────────────────────────

function startListening(guildId) {
  const state = guilds.get(guildId);
  if (!state?.connection) return;

  const connection = state.connection;
  const receiver = connection.receiver;

  const pendingTimers = new Map();
  const pendingChunks = new Map();

  let processing = false;
  const queue = [];

  const drainQueue = () => {
    if (processing || queue.length === 0) return;

    const job = queue.shift();
    processing = true;

    processAudio(job.guildId, job.userId, job.username, job.chunks)
      .catch((err) => console.error('[Queue] Error:', err.message))
      .finally(() => {
        processing = false;
        drainQueue();
      });
  };

  receiver.speaking.on('start', (userId) => {
    if (client.user && userId === client.user.id) return;

    if (pendingTimers.has(userId)) {
      clearTimeout(pendingTimers.get(userId));
      pendingTimers.delete(userId);
    }

    const member = client.guilds.cache.get(guildId)?.members.cache.get(userId);
    const username = member?.displayName || member?.user?.username || `User ${userId}`;

    const audioStream = receiver.subscribe(userId, {
      end: {
        behavior: EndBehaviorType.AfterSilence,
        duration: CFG.SILENCE_MS,
      },
    });

    const decoder = new prism.opus.Decoder({
      frameSize: FRAME_SIZE,
      channels: DISCORD_CHANNELS,
      rate: DISCORD_SAMPLE_RATE,
    });

    const chunks = [];

    audioStream.on('error', (err) => {
      console.error('[Audio Stream] Error:', err.message);
    });

    decoder.on('error', (err) => {
      console.error('[Decoder] Error:', err.message);
    });

    audioStream
      .pipe(decoder)
      .on('data', (chunk) => chunks.push(chunk))
      .on('end', () => {
        if (chunks.length === 0) return;

        const existing = pendingChunks.get(userId) || [];
        const merged = [...existing, ...chunks];
        pendingChunks.set(userId, merged);

        const timer = setTimeout(() => {
          pendingTimers.delete(userId);
          const finalChunks = pendingChunks.get(userId) || [];
          pendingChunks.delete(userId);
          if (finalChunks.length === 0) return;

          const job = { guildId, userId, username, chunks: finalChunks };

          if (!processing) {
            processing = true;
            processAudio(job.guildId, job.userId, job.username, job.chunks)
              .catch((err) => console.error('[Audio] Error:', err.message))
              .finally(() => {
                processing = false;
                drainQueue();
              });
          } else {
            queue.push(job);
          }
        }, CFG.DEBOUNCE_MS);

        pendingTimers.set(userId, timer);
      })
      .on('error', (err) => {
        console.error('[Pipe] Error:', err.message);
      });
  });

  receiver.speaking.on('error', (err) => {
    console.error('[Receiver] Error:', err.message);
  });
}

// ─────────────────────────────────────────────────────────────
// COMMANDS
// ─────────────────────────────────────────────────────────────

async function handleJoin(guildId, voiceChannel) {
  const state = ensureGuildState(guildId);
  if (state.connection) disconnectGuild(guildId);
  await connectToChannel(voiceChannel);
}

async function handleLeave(guildId) {
  disconnectGuild(guildId);
}

async function handleClear(guildId) {
  const state = ensureGuildState(guildId);
  state.history = [];
}

async function handleStatus(interactionOrReply, guildId) {
  const state = guilds.get(guildId);
  const model = process.env.OLLAMA_MODEL_OVERRIDE || CFG.OLLAMA_MODEL;

  const text = state
    ? [
      '**Status**',
      `Canal: **${state.channelName}**`,
      `Modelo: \`${model}\``,
      `Whisper: \`${CFG.WHISPER_MODEL}\``,
      `TTS: \`${CFG.TTS_ENGINE}\``,
      `RMS: \`${CFG.RMS_THRESHOLD}\``,
      `Silence: \`${CFG.SILENCE_MS}ms\``,
      `Debounce: \`${CFG.DEBOUNCE_MS}ms\``,
      `Histórico: ${state.history.length} mensagens`,
    ].join('\n')
    : [
      '**Status**',
      'Não conectado a nenhum canal.',
      `Modelo: \`${model}\``,
      `Whisper: \`${CFG.WHISPER_MODEL}\``,
      `TTS: \`${CFG.TTS_ENGINE}\``,
    ].join('\n');

  if (typeof interactionOrReply.reply === 'function') {
    await interactionOrReply.reply(text);
  } else if (typeof interactionOrReply.editReply === 'function') {
    await interactionOrReply.editReply(text);
  }
}

async function handleModelChange(interaction, guildId, newModel) {
  process.env.OLLAMA_MODEL_OVERRIDE = newModel;
  await interaction.reply(`Modelo trocado para: \`${newModel}\``);
}

async function handleSensitivity(interaction, value) {
  CFG.RMS_THRESHOLD = value;
  await interaction.reply(`RMS threshold ajustado para \`${value}\``);
}

async function handleSilence(interaction, value) {
  CFG.SILENCE_MS = value;
  await interaction.reply(`Silence ajustado para \`${value}ms\``);
}

async function executePrefixCommand(message) {
  const content = message.content.trim();
  if (!content.startsWith(CFG.PREFIX)) return;

  const args = content.slice(CFG.PREFIX.length).trim().split(/\s+/);
  const command = (args.shift() || '').toLowerCase();
  const guildId = message.guild.id;

  switch (command) {
    case 'join': {
      const voiceChannel = message.member?.voice?.channel;
      if (!voiceChannel) {
        await message.reply('Você precisa estar em um canal de voz primeiro.');
        return;
      }
      await handleJoin(guildId, voiceChannel);
      await message.reply(`Entrei em **${voiceChannel.name}**.`);
      break;
    }

    case 'leave': {
      await handleLeave(guildId);
      await message.reply('Desconectado.');
      break;
    }

    case 'clear': {
      await handleClear(guildId);
      await message.reply('Histórico limpo.');
      break;
    }

    case 'model': {
      const newModel = args.join(' ').trim();
      if (!newModel) {
        await message.reply(`Uso: ${CFG.PREFIX}model llama3.2:3b`);
        return;
      }
      process.env.OLLAMA_MODEL_OVERRIDE = newModel;
      await message.reply(`Modelo trocado para: \`${newModel}\``);
      break;
    }

    case 'status': {
      const state = guilds.get(guildId);
      const model = process.env.OLLAMA_MODEL_OVERRIDE || CFG.OLLAMA_MODEL;
      await message.reply(
        state
          ? `Canal: ${state.channelName}\nModelo: ${model}\nRMS: ${CFG.RMS_THRESHOLD}\nSilence: ${CFG.SILENCE_MS}ms`
          : `Não conectado.\nModelo: ${model}`
      );
      break;
    }

    case 'sensitivity': {
      const value = Number(args[0]);
      if (!Number.isFinite(value)) {
        await message.reply(`Uso: ${CFG.PREFIX}sensitivity 0.015`);
        return;
      }
      CFG.RMS_THRESHOLD = value;
      await message.reply(`RMS threshold ajustado para ${value}`);
      break;
    }

    case 'silence': {
      const value = Number(args[0]);
      if (!Number.isFinite(value)) {
        await message.reply(`Uso: ${CFG.PREFIX}silence 1000`);
        return;
      }
      CFG.SILENCE_MS = value;
      await message.reply(`Silence ajustado para ${value}ms`);
      break;
    }

    default:
      break;
  }
}

// ─────────────────────────────────────────────────────────────
// DISCORD EVENTS
// ─────────────────────────────────────────────────────────────

let bootstrapped = false;

async function onReady() {
  if (bootstrapped) return;
  bootstrapped = true;

  console.log(`[Bot] Logged in as ${client.user.tag}`);
  console.log('[Slash] Registering slash commands...');
  await registerSlashCommands();
  console.log('[Slash] Commands registered.');
  console.log(`[Bot] Model: ${CFG.OLLAMA_MODEL}`);
  console.log(`[Bot] Whisper: ${CFG.WHISPER_MODEL}`);
  console.log(`[Bot] Whisper lang: ${CFG.WHISPER_LANG}`);
  console.log(`[Bot] TTS: ${CFG.TTS_ENGINE}`);
  console.log(`[Bot] RMS threshold: ${CFG.RMS_THRESHOLD}`);
  console.log(`[Bot] Silence: ${CFG.SILENCE_MS}ms`);
  console.log(`[Bot] Debounce: ${CFG.DEBOUNCE_MS}ms`);

  if (CFG.AUTO_JOIN_CHANNEL_ID && CFG.GUILD_ID) {
    try {
      await client.guilds.fetch(CFG.GUILD_ID);
      const channel = await client.channels.fetch(CFG.AUTO_JOIN_CHANNEL_ID);
      if (channel?.isVoiceBased()) {
        await connectToChannel(channel);
      } else {
        console.error('[Auto-join] The configured channel is not voice-based.');
      }
    } catch (err) {
      console.error('[Auto-join] Failed:', err.message);
    }
  }
}

client.once('ready', onReady);
client.once('clientReady', onReady);

client.on('voiceStateUpdate', async (oldState, newState) => {
  if (!CFG.OWNER_USER_ID || newState.member?.id !== CFG.OWNER_USER_ID) return;
  const guildId = newState.guild.id;

  if (newState.channelId && newState.channelId !== oldState.channelId) {
    const existing = guilds.get(guildId);
    if (existing) disconnectGuild(guildId);

    try {
      await connectToChannel(newState.channel);
    } catch (err) {
      console.error('[Auto-follow] Failed:', err.message);
    }
  }

  if (!newState.channelId && oldState.channelId) {
    const state = guilds.get(guildId);
    if (state) {
      disconnectGuild(guildId);
      guilds.delete(guildId);
      console.log('[Bot] Owner left — disconnected.');
    }
  }
});

client.on('interactionCreate', async (interaction) => {
  try {
    if (!interaction.isChatInputCommand()) return;

    const { commandName, guildId } = interaction;

    switch (commandName) {
      case 'join': {
        const voiceChannel = interaction.member?.voice?.channel;
        if (!voiceChannel) {
          await interaction.reply({ content: 'You need to be in a voice channel first.', ephemeral: true });
          return;
        }
        await interaction.deferReply();
        try {
          await handleJoin(guildId, voiceChannel);
          await interaction.editReply(`Entrei em **${voiceChannel.name}**.`);
        } catch (err) {
          await interaction.editReply(`Falha ao entrar: ${err.message}`);
        }
        break;
      }

      case 'leave': {
        await handleLeave(guildId);
        await interaction.reply('Desconectado.');
        break;
      }

      case 'clear': {
        await handleClear(guildId);
        await interaction.reply('Histórico de conversa limpo.');
        break;
      }

      case 'model': {
        const newModel = interaction.options.getString('name', true);
        await handleModelChange(interaction, guildId, newModel);
        break;
      }

      case 'status': {
        await handleStatus(interaction, guildId);
        break;
      }

      case 'sensitivity': {
        const value = interaction.options.getNumber('value', true);
        await handleSensitivity(interaction, value);
        break;
      }

      case 'silence': {
        const value = interaction.options.getInteger('value', true);
        await handleSilence(interaction, value);
        break;
      }

      case 'debug': {
        await interaction.reply(
          `\`BOT_TOKEN\`: ${CFG.BOT_TOKEN ? 'set' : 'missing'}\n` +
          `\`GUILD_ID\`: ${CFG.GUILD_ID}\n` +
          `\`AUTO_JOIN_CHANNEL_ID\`: ${CFG.AUTO_JOIN_CHANNEL_ID || 'unset'}\n` +
          `\`OWNER_USER_ID\`: ${CFG.OWNER_USER_ID || 'unset'}\n` +
          `\`OLLAMA_MODEL\`: ${CFG.OLLAMA_MODEL}\n` +
          `\`WHISPER_MODEL\`: ${CFG.WHISPER_MODEL}\n` +
          `\`PIPER_MODEL\`: ${CFG.PIPER_MODEL}\n` +
          `\`TTS_ENGINE\`: ${CFG.TTS_ENGINE}\n` +
          `\`RMS_THRESHOLD\`: ${CFG.RMS_THRESHOLD}\n` +
          `\`MIN_AUDIO_MS\`: ${CFG.MIN_AUDIO_MS}\n` +
          `\`SILENCE_MS\`: ${CFG.SILENCE_MS}\n` +
          `\`DEBOUNCE_MS\`: ${CFG.DEBOUNCE_MS}`
        );
        break;
      }

      default:
        break;
    }
  } catch (err) {
    console.error('[Interaction] Error:', err.message);
    if (interaction.isRepliable() && !interaction.replied && !interaction.deferred) {
      await interaction.reply({ content: `Erro: ${err.message}`, ephemeral: true }).catch(() => {});
    }
  }
});

client.on('messageCreate', async (message) => {
  try {
    if (message.author.bot) return;
    if (!message.guild) return;
    await executePrefixCommand(message);
  } catch (err) {
    console.error('[Prefix] Error:', err.message);
  }
});

// ─────────────────────────────────────────────────────────────
// PROCESS HANDLERS
// ─────────────────────────────────────────────────────────────

process.on('unhandledRejection', (reason) => {
  console.error('[UnhandledRejection]', reason);
});

process.on('uncaughtException', (err) => {
  console.error('[UncaughtException]', err);
});

process.on('SIGINT', () => {
  console.log('\n[Bot] Shutting down...');
  for (const [, state] of guilds) {
    try {
      if (state.connection) state.connection.destroy();
    } catch {}
  }
  client.destroy();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n[Bot] Shutting down...');
  for (const [, state] of guilds) {
    try {
      if (state.connection) state.connection.destroy();
    } catch {}
  }
  client.destroy();
  process.exit(0);
});

// ─────────────────────────────────────────────────────────────
// START
// ─────────────────────────────────────────────────────────────

client.login(CFG.BOT_TOKEN).catch((err) => {
  console.error('[Login] Failed:', err.message);
  process.exit(1);
});
