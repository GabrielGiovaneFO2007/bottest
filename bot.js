/**
 * Discord Voice Agent Bot
 * Listens to voice channels, transcribes with whisper.cpp,
 * queries Ollama, and speaks responses with Kokoro TTS.
 *
 * Dependencies:
 *   npm install discord.js @discordjs/voice prism-media axios dotenv
 *   plus a voice backend for TTS (local Python env or a TTS HTTP server)
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
const { execFile } = require('child_process');
const axios = require('axios');

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────

const BOT_TOKEN = process.env.BOT_TOKEN;
const CLIENT_ID = process.env.CLIENT_ID;
const GUILD_ID = process.env.GUILD_ID;

const AUTO_JOIN_CHANNEL = process.env.AUTO_JOIN_CHANNEL_ID || null;
const OWNER_USER_ID = process.env.OWNER_USER_ID || null;

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434/api/chat';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3.2:3b';

const WHISPER_BIN =
  process.env.WHISPER_BIN ||
  findFirstExisting([
    path.join(os.homedir(), 'whisper.cpp/build/bin/whisper-cli'),
    path.join(os.homedir(), 'whisper.cpp/build/bin/main'),
    path.join(os.homedir(), 'whisper.cpp/main'),
  ]);

const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL ||
  path.join(os.homedir(), 'whisper.cpp/models/ggml-base.en.bin');

const WHISPER_LANG = process.env.WHISPER_LANG || 'pt';

const KOKORO_MODEL = process.env.KOKORO_MODEL || path.join(os.homedir(), 'kokoro-v1.0.onnx');
const KOKORO_VOICES = process.env.KOKORO_VOICES || path.join(os.homedir(), 'voices-v1.0.bin');
const TTS_SERVER_URL = process.env.TTS_SERVER_URL || ''; // optional, e.g. http://localhost:5500/tts
const TTS_PYTHON_BIN = process.env.TTS_PYTHON_BIN || 'python3';

const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT ||
  'Você é um assistente de voz útil em um canal do Discord. Responda de forma natural, clara e concisa. ' +
  'Evite markdown quando não for necessário. Se o usuário falar em português, responda em português.';

// Audio / speech tuning
const DISCORD_SAMPLE_RATE = 48000;
const DISCORD_CHANNELS = 2;
const WHISPER_SAMPLE_RATE = 16000;
const FRAME_SIZE = 960;

// Lower latency defaults
const SILENCE_DURATION_MS = Number(process.env.SILENCE_DURATION_MS || 1200);
const MIN_AUDIO_DURATION_MS = Number(process.env.MIN_AUDIO_DURATION_MS || 1200);
const DEBOUNCE_MS = Number(process.env.DEBOUNCE_MS || 300);

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
    .setDescription('Ajusta o limiar de ruído')
    .addNumberOption((opt) =>
      opt.setName('value')
        .setDescription('Maior = menos sensível')
        .setMinValue(0.01)
        .setMaxValue(0.2)
        .setRequired(true)
    ),
].map((cmd) => cmd.toJSON());

async function registerSlashCommands() {
  if (!CLIENT_ID || !GUILD_ID) {
    throw new Error('CLIENT_ID ou GUILD_ID não definidos no .env');
  }

  const rest = new REST({ version: '10' }).setToken(BOT_TOKEN);
  await rest.put(Routes.applicationGuildCommands(CLIENT_ID, GUILD_ID), { body: commands });
}

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────

// Per guild: { connection, player, history, channelName, busy }
const guilds = new Map();
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
// HELPERS
// ─────────────────────────────────────────────

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

function safeUnlink(filePath) {
  fs.unlink(filePath, () => {});
}

function normalizeSpeechText(text) {
  return String(text || '')
    .replace(/\*\*/g, '')
    .replace(/\*/g, '')
    .replace(/`/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractCompleteSentences(buffer, final = false) {
  const sentences = [];
  let rest = buffer.trim();

  while (true) {
    const match = rest.match(/^(.+?[.!?]+)(?=\s|$)/s);
    if (!match) break;

    const sentence = match[1].trim();
    if (sentence) sentences.push(sentence);

    rest = rest.slice(match[1].length).trimStart();
  }

  if (final && rest.trim()) {
    sentences.push(rest.trim());
    rest = '';
  }

  return { sentences, rest };
}

// ─────────────────────────────────────────────
// AUDIO UTILITIES
// ─────────────────────────────────────────────

function downsample(buffer) {
  const input = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  const ratio = DISCORD_SAMPLE_RATE / WHISPER_SAMPLE_RATE;
  const outLen = Math.floor(input.length / DISCORD_CHANNELS / ratio);
  const output = new Int16Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcIndex = Math.floor(i * ratio) * DISCORD_CHANNELS;
    output[i] = (input[srcIndex] + input[srcIndex + 1]) >> 1;
  }

  return Buffer.from(output.buffer);
}

function calcRms(buffer) {
  const samples = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i] / 32768;
    sum += s * s;
  }
  return Math.sqrt(sum / Math.max(samples.length, 1));
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

// ─────────────────────────────────────────────
// WHISPER
// ─────────────────────────────────────────────

function transcribe(wavPath) {
  return new Promise((resolve, reject) => {
    if (!WHISPER_BIN || !fs.existsSync(WHISPER_BIN)) {
      reject(new Error(`Whisper binary não encontrado: ${WHISPER_BIN}`));
      return;
    }

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

        resolve(text);
      }
    );
  });
}

// ─────────────────────────────────────────────
// OLLAMA
// ─────────────────────────────────────────────

async function queryOllamaNonStreaming(guildId, username, userText) {
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

    const reply = normalizeSpeechText(response.data?.message?.content || '');
    if (!reply) return null;

    state.history.push({ role: 'assistant', content: reply });
    if (state.history.length > 20) state.history = state.history.slice(-20);
    return reply;
  } catch (err) {
    console.error('[Ollama] Erro:', err.message);
    return 'Desculpe, não consegui acessar o modelo de IA agora.';
  }
}

async function streamOllamaAndQueueSpeech(guildId, username, userText, onSegmentReady) {
  const state = guilds.get(guildId);
  if (!state) return;

  const model = process.env.OLLAMA_MODEL_OVERRIDE || OLLAMA_MODEL;

  state.history.push({
    role: 'user',
    content: `[${username}]: ${userText}`,
  });

  let response;
  try {
    response = await axios.post(
      OLLAMA_URL,
      {
        model,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...state.history,
        ],
        stream: true,
      },
      { responseType: 'stream', timeout: 60000 }
    );
  } catch (err) {
    console.error('[Ollama] Streaming falhou:', err.message);
    const fallback = await queryOllamaNonStreaming(guildId, username, userText);
    if (fallback) await onSegmentReady(fallback, true);
    return;
  }

  let ndjsonBuffer = '';
  let assistantText = '';
  let buffer = '';
  let streamDone = false;

  const processBuffer = async (final = false) => {
    const { sentences, rest } = extractCompleteSentences(buffer, final);
    buffer = rest;

    for (const sentence of sentences) {
      const cleaned = normalizeSpeechText(sentence);
      if (!cleaned) continue;
      assistantText += (assistantText ? ' ' : '') + cleaned;
      await onSegmentReady(cleaned, false);
    }
  };

  await new Promise((resolve, reject) => {
    response.data.on('data', (chunk) => {
      ndjsonBuffer += chunk.toString('utf8');

      let nlIndex;
      while ((nlIndex = ndjsonBuffer.indexOf('\n')) !== -1) {
        const rawLine = ndjsonBuffer.slice(0, nlIndex).trim();
        ndjsonBuffer = ndjsonBuffer.slice(nlIndex + 1);
        if (!rawLine) continue;

        let obj;
        try {
          obj = JSON.parse(rawLine);
        } catch {
          continue;
        }

        const token = obj?.message?.content || '';
        if (token) {
          buffer += token;
        }

        if (obj?.done) {
          streamDone = true;
        }
      }

      // Process incrementally after each chunk.
      processBuffer(false).catch(reject);

      if (streamDone) {
        processBuffer(true)
          .then(() => resolve())
          .catch(reject);
      }
    });

    response.data.on('end', () => {
      processBuffer(true)
        .then(() => resolve())
        .catch(reject);
    });

    response.data.on('error', reject);
  });

  const reply = normalizeSpeechText(assistantText || buffer);
  const finalReply = reply || 'Desculpe, não consegui gerar uma resposta agora.';

  state.history.push({ role: 'assistant', content: finalReply });
  if (state.history.length > 20) state.history = state.history.slice(-20);
}

async function queryAndSpeak(guildId, username, userText, speakSegment) {
  await streamOllamaAndQueueSpeech(guildId, username, userText, speakSegment);
}

// ─────────────────────────────────────────────
// TTS
// ─────────────────────────────────────────────

async function generateTtsLocal(text, outputPath) {
  const safeText = String(text || '')
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, ' ')
    .trim();

  const asciiRatio = (safeText.match(/[a-zA-Z]/g) || []).length / Math.max(safeText.length, 1);
  const lang = asciiRatio > 0.85 ? 'en-us' : 'pt-br';
  const voice = lang === 'en-us' ? 'bf_emma' : 'pf_dora';

  const script = `
from kokoro_onnx import Kokoro
import soundfile as sf

k = Kokoro(${JSON.stringify(KOKORO_MODEL)}, ${JSON.stringify(KOKORO_VOICES)})
samples, sr = k.create(${JSON.stringify(safeText)}, voice=${JSON.stringify(voice)}, speed=1.0, lang=${JSON.stringify(lang)})
sf.write(${JSON.stringify(outputPath)}, samples, sr)
`;

  return new Promise((resolve, reject) => {
    execFile(TTS_PYTHON_BIN, ['-c', script], { timeout: 30000 }, (err, stdout, stderr) => {
      if (err) {
        reject(new Error(stderr || err.message));
        return;
      }
      resolve();
    });
  });
}

async function generateTtsViaServer(text, outputPath) {
  const asciiRatio = (text.match(/[a-zA-Z]/g) || []).length / Math.max(text.length, 1);
  const lang = asciiRatio > 0.85 ? 'en-us' : 'pt-br';

  const response = await axios.post(
    TTS_SERVER_URL,
    { text, lang },
    { responseType: 'arraybuffer', timeout: 20000 }
  );

  fs.writeFileSync(outputPath, Buffer.from(response.data));
}

async function generateTts(text, outputPath) {
  if (TTS_SERVER_URL) {
    return generateTtsViaServer(text, outputPath);
  }
  return generateTtsLocal(text, outputPath);
}

// ─────────────────────────────────────────────
// AUDIO PLAYER
// ─────────────────────────────────────────────

function playAudio(guildId, audioPath) {
  const state = guilds.get(guildId);
  if (!state?.player) return Promise.resolve();

  return new Promise((resolve) => {
    const stream = fs.createReadStream(audioPath);
    const resource = createAudioResource(stream, { inputType: StreamType.Arbitrary });

    const done = () => {
      state.player.off('error', onError);
      resolve();
    };

    const onError = (err) => {
      console.error('[Player] Erro:', err.message);
      done();
    };

    state.player.once(AudioPlayerStatus.Idle, done);
    state.player.once('error', onError);
    state.player.play(resource);
  });
}

// ─────────────────────────────────────────────
// PIPELINE
// ─────────────────────────────────────────────

async function processAudio(guildId, userId, username, pcmChunks) {
  const tmpDir = os.tmpdir();
  const wavPath = path.join(tmpDir, `discord_${guildId}_${userId}_${Date.now()}.wav`);

  try {
    const rawPcm = Buffer.concat(pcmChunks);
    const pcm16khz = downsample(rawPcm);

    const durationMs = (pcm16khz.length / 2 / WHISPER_SAMPLE_RATE) * 1000;
    if (durationMs < MIN_AUDIO_DURATION_MS) {
      console.log(`[Filter] Ignorado: ${Math.round(durationMs)}ms (abaixo do mínimo)`);
      return;
    }

    const rms = calcRms(pcm16khz);
    if (rms < currentThreshold) {
      console.log(`[Filter] Ignorado: RMS ${rms.toFixed(4)} abaixo do threshold ${currentThreshold}`);
      return;
    }

    await writePcmToWav(pcm16khz, wavPath);
    console.log(`[Whisper] Transcrevendo ${username} (${Math.round(durationMs)}ms, RMS ${rms.toFixed(4)})...`);

    const text = normalizeSpeechText(await transcribe(wavPath));
    if (!text || text.length < 3) {
      console.log('[Whisper] Transcrição vazia, ignorando.');
      return;
    }

    console.log(`[${username}]: ${text}`);

    const state = guilds.get(guildId);
    if (!state) return;

    const ttsQueue = [];
    let ttsRunning = false;
    let streamFinished = false;
    let resolveDrain;
    const drainPromise = new Promise((resolve) => {
      resolveDrain = resolve;
    });

    const maybeResolveDrain = () => {
      if (streamFinished && !ttsRunning && ttsQueue.length === 0) {
        resolveDrain();
      }
    };

    const pumpTtsQueue = async () => {
      if (ttsRunning) return;
      ttsRunning = true;

      while (ttsQueue.length > 0) {
        const segment = normalizeSpeechText(ttsQueue.shift());
        if (!segment || segment.length < 2) continue;

        const ttsPath = path.join(tmpDir, `tts_${guildId}_${Date.now()}_${Math.random().toString(16).slice(2)}.wav`);
        try {
          console.log(`[Bot/TTS] ${segment}`);
          await generateTts(segment, ttsPath);
          await playAudio(guildId, ttsPath);
        } catch (err) {
          console.error('[TTS] Erro:', err.message);
        } finally {
          safeUnlink(ttsPath);
        }
      }

      ttsRunning = false;
      maybeResolveDrain();
    };

    const enqueueSegment = async (segment, final = false) => {
      const cleaned = normalizeSpeechText(segment);
      if (cleaned) {
        ttsQueue.push(cleaned);
        void pumpTtsQueue();
      }
      if (final) {
        streamFinished = true;
        maybeResolveDrain();
      }
    };

    console.log('[Ollama] Streaming...');
    await queryAndSpeak(guildId, username, text, enqueueSegment);

    streamFinished = true;
    maybeResolveDrain();
    await drainPromise;
  } catch (err) {
    console.error('[Pipeline] Erro:', err.message);
  } finally {
    safeUnlink(wavPath);
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
// AUDIO RECEIVER
// ─────────────────────────────────────────────

function startListening(guildId) {
  const state = guilds.get(guildId);
  if (!state) return;

  const { connection } = state;
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
      .catch((err) => console.error('[Queue] Erro:', err.message))
      .finally(() => {
        processing = false;
        drainQueue();
      });
  };

  receiver.speaking.on('start', (userId) => {
    if (userId === client.user?.id) return;

    if (pendingTimers.has(userId)) {
      clearTimeout(pendingTimers.get(userId));
      pendingTimers.delete(userId);
    }

    const member = client.guilds.cache.get(guildId)?.members.cache.get(userId);
    const username = member?.displayName || member?.user?.username || `User ${userId}`;

    const audioStream = receiver.subscribe(userId, {
      end: {
        behavior: EndBehaviorType.AfterSilence,
        duration: SILENCE_DURATION_MS,
      },
    });

    const decoder = new prism.opus.Decoder({
      frameSize: FRAME_SIZE,
      channels: DISCORD_CHANNELS,
      rate: DISCORD_SAMPLE_RATE,
    });

    const chunks = [];

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

          if (!processing) {
            processing = true;
            processAudio(guildId, userId, username, finalChunks)
              .catch((err) => console.error('[Audio] Erro:', err.message))
              .finally(() => {
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

let bootstrapped = false;

async function onClientReady() {
  if (bootstrapped) return;
  bootstrapped = true;

  console.log(`\n[Bot] Logado como ${client.user.tag}`);
  console.log('[Slash] Registrando comandos...');
  await registerSlashCommands();
  console.log('[Slash] Comandos registrados.');
  console.log(`[Bot] Modelo: ${OLLAMA_MODEL}`);
  console.log(`[Bot] Whisper: ${WHISPER_MODEL_PATH}`);
  console.log(`[Bot] Idioma: ${WHISPER_LANG}`);
  console.log(`[Bot] Threshold inicial: ${currentThreshold}`);
  console.log(`[Bot] Silence duration: ${SILENCE_DURATION_MS}ms`);
  console.log(`[Bot] Debounce: ${DEBOUNCE_MS}ms`);

  if (AUTO_JOIN_CHANNEL && GUILD_ID) {
    try {
      await client.guilds.fetch(GUILD_ID);
      const channel = await client.channels.fetch(AUTO_JOIN_CHANNEL);
      if (channel?.isVoiceBased()) {
        await connectToChannel(channel);
      } else {
        console.error('[Auto-join] O canal informado não é um canal de voz.');
      }
    } catch (err) {
      console.error('[Auto-join] Falhou:', err.message);
    }
  }
}

client.once('ready', onClientReady);
client.once('clientReady', onClientReady);

client.on('voiceStateUpdate', async (oldState, newState) => {
  if (!OWNER_USER_ID || newState.member?.id !== OWNER_USER_ID) return;
  const guildId = newState.guild.id;

  if (newState.channelId && newState.channelId !== oldState.channelId) {
    const existing = guilds.get(guildId);
    if (existing) {
      existing.connection.destroy();
      guilds.delete(guildId);
    }

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
      if (existing) {
        existing.connection.destroy();
        guilds.delete(guildId);
      }

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
  for (const [, state] of guilds) {
    try {
      state.connection.destroy();
    } catch {
      // ignore
    }
  }
  client.destroy();
  process.exit(0);
});
