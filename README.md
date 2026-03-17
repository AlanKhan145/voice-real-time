# realtime-subtitle

Local-only realtime speech → subtitle → speaker attribution → translation pipeline.

No web UI. No cloud. No databases. Console output only.

```
[partial] xin chào mọi người hôm nay
[partial] xin chào mọi người hôm nay chúng ta bắt đầu
[final][seg_3f2a1b] Xin chào mọi người, hôm nay chúng ta bắt đầu.
[speaker][seg_3f2a1b] speaker_1  (conf 91%)
[translation][seg_3f2a1b] Hello everyone, today we begin.
[enriched][seg_3f2a1b][speaker_1]
  SRC : Xin chào mọi người, hôm nay chúng ta bắt đầu.
  TRX : Hello everyone, today we begin.
```

---

## Architecture

```
Microphone
  └─► RealtimeSTT (VAD + streaming Whisper)
        ├─► on_partial  ──► TextStabilizer ──► PartialUpdatedEvent ──► ConsoleRenderer
        ├─► on_stabilized ──────────────────► StabilizedUpdatedEvent ──► ConsoleRenderer
        └─► on_final ──► SentenceBoundaryDetector ──► SentenceFinalizedEvent
                                                             │
                                         ┌───────────────────┴──────────────────┐
                                         ▼                                      ▼
                                  SpeakerWorker                       TranslationWorker
                                  (thread pool)                        (thread pool)
                                         │                                      │
                              SpeakerCompletedEvent              TranslationCompletedEvent
                                         └───────────────────┬──────────────────┘
                                                             ▼
                                                       PatchUpdater
                                                             │
                                                    SegmentEnrichedEvent
                                                             │
                                                      ConsoleRenderer
```

**Hot path** (partial subtitles) is never blocked by speaker/translation workers.  
Workers run in separate thread pools and publish results back through the asyncio event bus.

---

## Project structure

```
realtime-subtitle/
├── app/
│   ├── main.py              # Entry point, CLI arg parsing, asyncio runner
│   ├── config.py            # Config dataclass loaded from env / .env
│   └── logging_config.py    # Structured logging setup
├── core/
│   ├── models.py            # Segment, AudioSpanRef, all Event types
│   ├── event_bus.py         # asyncio.Queue-based pub/sub event bus
│   ├── session_manager.py   # In-memory segment store (thread-safe)
│   └── patch_updater.py     # Merges worker results into segments
├── asr/
│   ├── realtime_recorder.py # RealtimeSTT wrapper, audio buffering
│   ├── stabilizer.py        # Partial text debouncer (anti-flicker)
│   └── sentence_boundary.py # Sentence-end heuristics
├── workers/
│   ├── speaker_worker.py    # Runs speaker identification off event loop
│   └── translation_worker.py# Runs translation off event loop
├── providers/
│   ├── speaker/
│   │   ├── base.py          # SpeakerProvider ABC
│   │   └── local_provider.py# resemblyzer → heuristic fallback
│   └── translation/
│       ├── base.py          # TranslationProvider ABC
│       └── local_provider.py# argostranslate → noop fallback
├── cli/
│   └── console_renderer.py  # All stdout rendering
├── scripts/
│   └── install_argos_packages.py
├── tests/
│   ├── test_event_bus.py
│   ├── test_session_manager.py
│   ├── test_patch_updater.py
│   └── test_enrichment_flow.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick start (local, venv)

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt install portaudio19-dev libsndfile1 ffmpeg

# Arch
sudo pacman -S portaudio libsndfile ffmpeg
```

### 2. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3. Optional: install speaker & translation backends

```bash
# Speaker diarization (adds resemblyzer, ~50 MB)
pip install resemblyzer

# Offline translation (adds argostranslate + ctranslate2, ~200 MB)
pip install argostranslate

# Download language packages (vi↔en by default)
python scripts/install_argos_packages.py

# Or a specific pair:
python scripts/install_argos_packages.py --from vi --to en
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env as needed — all values have sensible defaults
```

### 5. Run

```bash
python -m app.main
```

Common overrides:

```bash
# English source, translate to Vietnamese
python -m app.main --language en --target-language vi

# Use large model on GPU
python -m app.main --model-size large-v3 --device cuda

# Debug logging, no speaker, no translation
python -m app.main --log-level DEBUG --no-speaker --no-translation

# Tiny model — fastest startup, lower accuracy
python -m app.main --model-size tiny
```

---

## Configuration reference

All env vars (also accepted in `.env`):

| Variable | Default | Description |
|---|---|---|
| `LANGUAGE` | `vi` | Source language (BCP-47) |
| `TARGET_LANGUAGE` | `en` | Translation target language |
| `MODEL_SIZE` | `base` | Whisper model: tiny/base/small/medium/large-v3 |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `COMPUTE_TYPE` | `int8` | `int8` (CPU) / `float16` (GPU) / `float32` |
| `REALTIME_MODEL_SIZE` | `tiny` | Model for realtime partial transcription |
| `REALTIME_PROCESSING_PAUSE` | `0.2` | Seconds between partial updates |
| `SILERO_SENSITIVITY` | `0.4` | VAD sensitivity (0–1) |
| `WEBRTC_SENSITIVITY` | `3` | WebRTC VAD aggressiveness (0–3) |
| `POST_SPEECH_SILENCE_DURATION` | `0.7` | Seconds of silence → sentence end |
| `MIN_LENGTH_OF_RECORDING` | `0.5` | Minimum recording length in seconds |
| `SPEAKER_ENABLED` | `true` | Enable speaker attribution |
| `SPEAKER_PROVIDER` | `local` | `local` or `noop` |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | resemblyzer cosine sim threshold |
| `TRANSLATION_ENABLED` | `true` | Enable translation |
| `TRANSLATION_PROVIDER` | `argos` | `argos` or `noop` |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FILE` | _(none)_ | Write logs to this file as well |
| `WORKER_MAX_THREADS` | `4` | Thread pool size (split between workers) |

---

## Docker

### CPU mode

```bash
docker compose up
```

### GPU mode (NVIDIA)

```bash
docker compose --profile gpu up
```

### Microphone passthrough

The compose file uses `/dev/snd` (ALSA). If you use PulseAudio, uncomment the
PulseAudio section in `docker-compose.yml`:

```yaml
environment:
  PULSE_SERVER: unix:${XDG_RUNTIME_DIR}/pulse/native
volumes:
  - ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native
```

### Build only

```bash
docker build -t realtime-subtitle .
```

### Override config at runtime

```bash
docker run --rm -it \
  --device /dev/snd \
  --group-add audio \
  -e LANGUAGE=en \
  -e TARGET_LANGUAGE=vi \
  -e MODEL_SIZE=small \
  realtime-subtitle
```

---

## Tests

```bash
pytest tests/ -v
```

The test suite uses **fake providers** so no microphone or models are required.

---

## Extending backends

### Swap speaker provider

Implement `providers/speaker/base.py:SpeakerProvider`:

```python
from providers.speaker.base import SpeakerProvider, SpeakerResult
from core.models import AudioSpanRef

class MyCloudSpeaker(SpeakerProvider):
    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        # call your API here
        return SpeakerResult(speaker_id="...", confidence=0.9, provider="cloud")
```

Then inject it in `app/main.py:_build_speaker_provider()`.

### Swap translation provider

Implement `providers/translation/base.py:TranslationProvider`:

```python
from providers.translation.base import TranslationProvider, TranslationResult

class MyProvider(TranslationProvider):
    def translate(self, text, source_lang, target_lang) -> TranslationResult:
        ...
```

Then register it in `providers/translation/local_provider.py:create_translation_provider()`.

### Replace event bus with Redis Streams / Kafka

The `EventBus` class exposes only:
- `subscribe(event_type, handler)` 
- `publish(event)` (async)
- `publish_threadsafe(event)` (thread-safe sync)
- `run()` (async dispatcher loop)

Implement these four methods in a `RedisEventBus` / `KafkaEventBus` class and
swap in `app/main.py`. No other file changes needed.

---

## Model download locations

| Tool | Cache location |
|---|---|
| Whisper (faster-whisper) | `~/.cache/huggingface/hub/` |
| argostranslate | `~/.local/share/argos-translate/` |
| resemblyzer | `~/.cache/torch/` |

---

## Troubleshooting

**No audio input / PortAudio error**  
→ Check `portaudio19-dev` is installed. Verify microphone: `arecord -l`

**Whisper model download on first run**  
→ Normal. `base` downloads ~150 MB once. Set `MODEL_SIZE=tiny` for faster startup.

**argostranslate: no package for vi→en**  
→ Run `python scripts/install_argos_packages.py`

**Speaker always "speaker_unknown"**  
→ `resemblyzer` not installed or audio span is too short. Install resemblyzer or
increase `MIN_LENGTH_OF_RECORDING`.

**Partial subtitles flicker**  
→ Increase `REALTIME_PROCESSING_PAUSE` (e.g. `0.4`) or tune stabilizer constants
in `asr/stabilizer.py`.

**Sentence cuts too early**  
→ Increase `POST_SPEECH_SILENCE_DURATION` (e.g. `1.2`) to wait longer before finalizing.
# voice-real-time
