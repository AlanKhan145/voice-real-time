# realtime-subtitle

> Pipeline phụ đề realtime chạy hoàn toàn local — không cloud, không UI, không database.

Chuyển giọng nói từ microphone thành phụ đề realtime, tự động gán người nói, và dịch câu hoàn chỉnh — tất cả song song, trên máy tính của bạn.

```
────────────────────────────────────────────────────────────
 realtime-subtitle  |  local speech pipeline
 session: session_3f2a1b4c
────────────────────────────────────────────────────────────
[recorder] listening …
[partial] xin chào mọi người hôm nay
[partial] xin chào mọi người hôm nay chúng ta bắt đầu
[stabilized] xin chào mọi người, hôm nay chúng ta bắt đầu
[final][seg_7c3d0e] Xin chào mọi người, hôm nay chúng ta bắt đầu.
[speaker][seg_7c3d0e] speaker_1  (conf 91%)
[translation][seg_7c3d0e] Hello everyone, today we begin.
[enriched][seg_7c3d0e][speaker_1]
  SRC : Xin chào mọi người, hôm nay chúng ta bắt đầu.
  TRX : Hello everyone, today we begin.
```

---

## Mục lục

- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt nhanh](#cài-đặt-nhanh)
- [Chạy chương trình](#chạy-chương-trình)
- [Cấu hình](#cấu-hình)
- [Kiến trúc](#kiến-trúc)
- [Cấu trúc project](#cấu-trúc-project)
- [Segment data model](#segment-data-model)
- [Chọn model phù hợp](#chọn-model-phù-hợp)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [Test](#test)
- [Mở rộng backend](#mở-rộng-backend)
- [Troubleshooting](#troubleshooting)
- [Câu hỏi thường gặp](#câu-hỏi-thường-gặp)

---

## Tính năng

| Tính năng | Mô tả |
|---|---|
| Phụ đề realtime | In từng từ khi người dùng đang nói (hot path không bị block) |
| Stabilized output | Khử nhấp nháy — chỉ cập nhật khi text thay đổi đáng kể |
| Final sentence | Phát hiện câu hoàn chỉnh qua VAD + silence heuristic |
| Speaker attribution | Gán `speaker_1`, `speaker_2`… dựa trên voice embedding (resemblyzer) hoặc heuristic |
| Offline translation | Dịch toàn câu sau khi finalize, dùng argostranslate (hoàn toàn local) |
| Event-driven | Tất cả component giao tiếp qua internal event bus — dễ thay thế từng phần |
| Graceful shutdown | Ctrl+C → chờ worker xong → in tổng kết session |
| Zero cloud | Không cần internet sau khi tải model lần đầu |

---

## Yêu cầu hệ thống

### Phần cứng

| Chế độ | RAM tối thiểu | Ghi chú |
|---|---|---|
| CPU — model `tiny` | 2 GB | Độ trễ ~1–2 giây |
| CPU — model `base` | 4 GB | Khuyến nghị cho độ chính xác tốt |
| CPU — model `small` | 6 GB | Chất lượng cao hơn, chậm hơn |
| GPU (NVIDIA) | 4 GB VRAM | float16, tốc độ tốt nhất |

Microphone bất kỳ được hệ điều hành nhận diện đều hoạt động.

### Phần mềm

- **Python 3.11+**
- **Linux** (Ubuntu 20.04+, Debian, Arch, …) — macOS có thể hoạt động nhưng chưa được kiểm tra kỹ
- `portaudio19-dev`, `libsndfile1`, `ffmpeg` (xem bên dưới)
- Docker + Docker Compose (nếu chạy bằng container)

---

## Cài đặt nhanh

### Bước 1 — System dependencies

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y portaudio19-dev libsndfile1 ffmpeg

# Arch Linux
sudo pacman -S portaudio libsndfile ffmpeg

# Fedora / RHEL
sudo dnf install portaudio-devel libsndfile ffmpeg
```

### Bước 2 — Python virtual environment

```bash
git clone <repo-url> realtime-subtitle
cd realtime-subtitle

python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

> **Lần đầu chạy**, faster-whisper sẽ tự tải model Whisper (`base` ≈ 150 MB) về
> `~/.cache/huggingface/`. Kết nối internet cần thiết **chỉ trong lần đầu**.
>
> **Lưu ý:** Project dùng `sounddevice` + `webrtcvad` thay cho RealtimeSTT (PyAudio)
> để tránh lỗi build trên Linux. Không cần cài `portaudio19-dev` hay `python3-dev`.

### Bước 3 — (Khuyến nghị) Cài speaker + translation backend

```bash
# Speaker diarization — voice embedding cục bộ (~50 MB)
pip install resemblyzer

# Offline translation — dịch không cần internet (~200 MB + model ngôn ngữ)
pip install argostranslate

# Tải gói ngôn ngữ vi ↔ en (mặc định)
python scripts/install_argos_packages.py

# Hoặc chỉ một chiều cụ thể
python scripts/install_argos_packages.py --from vi --to en
python scripts/install_argos_packages.py --from en --to vi
```

> Nếu bỏ qua bước này, speaker sẽ dùng heuristic đơn giản và translation sẽ
> trả về text gốc với nhãn `[noop]`.

### Bước 4 — Cấu hình

```bash
cp .env.example .env
# Chỉnh sửa .env nếu cần — mặc định đã sẵn sàng chạy tiếng Việt
```

---

## Chạy chương trình

### Lệnh cơ bản

```bash
python -m app.main
```

### Các tùy chọn CLI phổ biến

```bash
# Nguồn tiếng Anh, dịch sang tiếng Việt
python -m app.main --language en --target-language vi

# Dùng model lớn hơn (chính xác hơn, chậm khởi động hơn)
python -m app.main --model-size small

# Chạy trên GPU NVIDIA
python -m app.main --device cuda --model-size medium

# Chỉ xem phụ đề, tắt speaker và dịch
python -m app.main --no-speaker --no-translation

# Debug — xem toàn bộ log nội bộ
python -m app.main --log-level DEBUG

# Dùng noop provider (hữu ích khi test pipeline mà không cần model thật)
python -m app.main --speaker-provider noop --translation-provider noop
```

### Thoát chương trình

Nhấn `Ctrl+C` — chương trình sẽ:
1. Dừng microphone
2. Chờ các worker đang xử lý hoàn tất (tối đa ~1.5 giây)
3. In tổng kết session (số segment, speaker, translation của từng câu)
4. Thoát sạch

---

## Cấu hình

Tất cả biến môi trường đều có thể đặt trong file `.env` hoặc export trực tiếp.
CLI flags sẽ ghi đè env nếu được cung cấp.

### ASR

| Biến | Mặc định | Mô tả |
|---|---|---|
| `LANGUAGE` | `vi` | Ngôn ngữ nguồn (BCP-47: `vi`, `en`, `zh`, `fr`, …) |
| `MODEL_SIZE` | `base` | Kích thước model Whisper (xem [bảng so sánh](#chọn-model-phù-hợp)) |
| `DEVICE` | `cpu` | `cpu` hoặc `cuda` |
| `COMPUTE_TYPE` | `int8` | `int8` (CPU nhanh nhất) / `float16` (GPU) / `float32` |
| `REALTIME_MODEL_SIZE` | `tiny` | Model dùng cho partial realtime (khuyến nghị `tiny` để tốc độ) |
| `REALTIME_PROCESSING_PAUSE` | `0.2` | Giây giữa mỗi lần cập nhật partial text |

### VAD / Phát hiện khoảng lặng

| Biến | Mặc định | Mô tả |
|---|---|---|
| `SILERO_SENSITIVITY` | `0.4` | Độ nhạy VAD silero (0.0–1.0, cao = nhạy hơn) |
| `WEBRTC_SENSITIVITY` | `3` | Độ mạnh WebRTC VAD (0–3) |
| `POST_SPEECH_SILENCE_DURATION` | `0.7` | Giây im lặng để xác định câu đã kết thúc |
| `MIN_LENGTH_OF_RECORDING` | `0.5` | Độ dài tối thiểu (giây) để được xem là câu hợp lệ |
| `MIN_GAP_BETWEEN_RECORDINGS` | `0.01` | Khoảng cách tối thiểu giữa hai lần ghi |

> **Nói liên tục bị cắt câu?** Tăng `POST_SPEECH_SILENCE_DURATION` lên `1.0–1.5`.  
> **Phụ đề chậm phản hồi?** Giảm `POST_SPEECH_SILENCE_DURATION` xuống `0.4–0.5`.

### Speaker attribution

| Biến | Mặc định | Mô tả |
|---|---|---|
| `SPEAKER_ENABLED` | `true` | Bật/tắt speaker attribution |
| `SPEAKER_PROVIDER` | `local` | `local` (resemblyzer → heuristic) hoặc `noop` |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity tối thiểu để nhận diện cùng speaker |

> Giảm `SPEAKER_SIMILARITY_THRESHOLD` (vd: `0.65`) nếu cùng người bị label thành nhiều speaker khác nhau.  
> Tăng lên (vd: `0.85`) nếu nhiều người bị gộp chung một label.

### Translation

| Biến | Mặc định | Mô tả |
|---|---|---|
| `TRANSLATION_ENABLED` | `true` | Bật/tắt dịch |
| `TRANSLATION_PROVIDER` | `argos` | `argos` (offline) hoặc `noop` |
| `TARGET_LANGUAGE` | `en` | Ngôn ngữ đích |

### Workers & logging

| Biến | Mặc định | Mô tả |
|---|---|---|
| `WORKER_MAX_THREADS` | `4` | Tổng số thread cho speaker + translation pool |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FILE` | _(trống)_ | Ghi log song song vào file này |

---

## Kiến trúc

### Hai lane độc lập

```
Microphone
  └─► RealtimeSTT (VAD + streaming Whisper)
        │
        ├─[realtime update]─► TextStabilizer ──► PartialUpdatedEvent
        │                                                │
        ├─[stabilized]──────────────────────► StabilizedUpdatedEvent
        │                                                │
        └─[VAD silence]─► SentenceBoundaryDetector       │
                                  │                      │
                        SentenceFinalizedEvent     ConsoleRenderer
                                  │
           ┌──────────────────────┴───────────────────────┐
           ▼                                              ▼
    SpeakerWorker                               TranslationWorker
    (ThreadPoolExecutor)                        (ThreadPoolExecutor)
           │                                              │
    SpeakerCompletedEvent                  TranslationCompletedEvent
           └──────────────────────┬───────────────────────┘
                                  ▼
                           PatchUpdater
                      (merge vào Segment)
                                  │
                         SegmentEnrichedEvent
                                  │
                           ConsoleRenderer
```

**Hot path** (partial → stabilized → final) **không bao giờ bị block** bởi speaker hay translation.  
Worker chạy trong thread pool riêng, kết quả được publish lại vào event bus async.

### Event bus

```
EventBus (asyncio.Queue)
  ├── publish(event)          → async, từ trong event loop
  ├── publish_threadsafe(ev)  → sync, từ bất kỳ thread (RealtimeSTT callback)
  ├── subscribe(type, handler)→ đăng ký async handler
  └── run()                   → dispatcher coroutine, chạy trong main event loop
```

Interface này đủ mỏng để thay thế bằng `RedisEventBus` hoặc `KafkaEventBus` mà không cần đổi subscriber.

### Vòng đời của một Segment

```
[partial_updated] ×N          — nhiều lần, hot path
[stabilized_updated] ×M       — mỗi khi text ổn định đủ
[sentence_finalized]          — 1 lần, tạo Segment trong SessionManager
    ├── [speaker_completed]   — async, có thể đến trước hoặc sau translation
    └── [translation_completed]
[segment_enriched]            — khi cả hai worker hoàn tất
```

---

## Cấu trúc project

```
realtime-subtitle/
│
├── app/
│   ├── main.py              # Entry point: CLI args → asyncio.run(run(config))
│   ├── config.py            # Config dataclass, load từ env / .env
│   └── logging_config.py    # Cấu hình logging có cấu trúc
│
├── core/
│   ├── models.py            # Segment, AudioSpanRef, tất cả Event dataclass
│   ├── event_bus.py         # Pub/sub trên asyncio.Queue
│   ├── session_manager.py   # In-memory store thread-safe cho Segment
│   └── patch_updater.py     # Nhận kết quả worker, patch Segment, phát enriched
│
├── asr/
│   ├── realtime_recorder.py # Wrap RealtimeSTT, buffer audio, bridge thread→asyncio
│   ├── stabilizer.py        # Debounce partial text, chống nhấp nháy
│   └── sentence_boundary.py # Heuristic phát hiện câu hoàn chỉnh
│
├── workers/
│   ├── speaker_worker.py    # Lắng nghe sentence_finalized, chạy provider ngoài event loop
│   └── translation_worker.py# Như trên, cho translation
│
├── providers/
│   ├── speaker/
│   │   ├── base.py          # SpeakerProvider ABC + SpeakerResult
│   │   └── local_provider.py# ResemblyzerBackend → HeuristicBackend → NoopSpeakerProvider
│   └── translation/
│       ├── base.py          # TranslationProvider ABC + TranslationResult
│       └── local_provider.py# ArgosTranslationProvider → NoopTranslationProvider
│
├── cli/
│   └── console_renderer.py  # Tất cả stdout rendering, ANSI colour, in-place partial
│
├── scripts/
│   └── install_argos_packages.py  # Tải gói ngôn ngữ argostranslate
│
├── tests/
│   ├── conftest.py
│   ├── test_event_bus.py          # publish, subscribe, thread-safe, exception isolation
│   ├── test_session_manager.py    # create, patch, enrichment state machine
│   ├── test_patch_updater.py      # speaker + translation patch, enriched trigger
│   └── test_enrichment_flow.py    # E2E với fake providers, không cần mic/model
│
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Segment data model

Mỗi câu được lưu thành một `Segment`:

```python
@dataclass
class Segment:
    session_id: str           # ID phiên làm việc
    segment_id: str           # ID duy nhất của câu (vd: seg_3f2a1b)
    start_ms: int             # Thời điểm bắt đầu ghi (ms, monotonic clock)
    end_ms: int               # Thời điểm kết thúc ghi

    source_text_partial: str  # Text đang cập nhật realtime
    source_text_final: str    # Text hoàn chỉnh sau VAD
    is_final: bool

    speaker_status: WorkerStatus  # pending | done | failed
    speaker_id: Optional[str]     # "speaker_1", "speaker_2", …
    speaker_confidence: float     # 0.0–1.0

    translation_status: WorkerStatus
    translated_text: Optional[str]

    audio_span_ref: Optional[AudioSpanRef]  # PCM bytes cho speaker worker
    created_at: datetime
    updated_at: datetime
```

`Segment.is_enriched` trả về `True` khi cả hai worker đã về trạng thái `done` hoặc `failed`.

---

## Chọn model phù hợp

### Model Whisper cho final transcription (`MODEL_SIZE`)

| Model | RAM | Tốc độ (CPU) | Độ chính xác | Khuyến nghị |
|---|---|---|---|---|
| `tiny` | ~1 GB | Rất nhanh | Thấp | Kiểm thử nhanh, máy yếu |
| `base` | ~1.5 GB | Nhanh | Tốt | **Mặc định — cân bằng tốt** |
| `small` | ~2.5 GB | Vừa | Khá cao | Dùng hàng ngày trên CPU đủ mạnh |
| `medium` | ~5 GB | Chậm (CPU) | Cao | Khuyến nghị với GPU |
| `large-v3` | ~10 GB | Rất chậm (CPU) | Rất cao | Chỉ nên dùng với GPU |

### Model cho realtime partial (`REALTIME_MODEL_SIZE`)

Nên để `tiny` để partial cập nhật nhanh. Chất lượng partial không cần cao vì chỉ dùng để hiển thị tạm thời — `source_text_final` mới là kết quả chính xác.

### Gợi ý cấu hình theo máy

| Thiết bị | Cấu hình |
|---|---|
| Laptop thông thường (CPU 4 nhân, 8 GB RAM) | `MODEL_SIZE=base`, `DEVICE=cpu`, `COMPUTE_TYPE=int8` |
| Máy tính để bàn mạnh (CPU 8+ nhân) | `MODEL_SIZE=small`, `DEVICE=cpu` |
| GPU NVIDIA 4 GB VRAM | `MODEL_SIZE=medium`, `DEVICE=cuda`, `COMPUTE_TYPE=float16` |
| GPU NVIDIA 8+ GB VRAM | `MODEL_SIZE=large-v3`, `DEVICE=cuda`, `COMPUTE_TYPE=float16` |

---

## Chạy bằng Docker

### CPU (mặc định)

```bash
# Build + chạy
docker compose up

# Chỉ build
docker build -t realtime-subtitle .
```

### GPU (NVIDIA)

Yêu cầu [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose --profile gpu up
```

### Microphone passthrough

**ALSA** (mặc định trong `docker-compose.yml`):
```yaml
devices:
  - /dev/snd:/dev/snd
group_add:
  - audio
```

**PulseAudio** — uncomment phần này trong `docker-compose.yml`:
```yaml
environment:
  PULSE_SERVER: unix:${XDG_RUNTIME_DIR}/pulse/native
volumes:
  - ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native
group_add:
  - audio
```

### Ghi đè config khi chạy

```bash
docker run --rm -it \
  --device /dev/snd \
  --group-add audio \
  -e LANGUAGE=en \
  -e TARGET_LANGUAGE=vi \
  -e MODEL_SIZE=small \
  -e SPEAKER_PROVIDER=noop \
  realtime-subtitle
```

### Volumes — model cache

Docker Compose tạo sẵn 2 named volumes để model không bị tải lại mỗi lần:

```
whisper-models  → /root/.cache/huggingface
argos-models    → /root/.local/share/argos-translate
```

---

## Test

```bash
# Chạy toàn bộ test suite (không cần mic, không cần model)
pytest tests/ -v

# Chỉ một file
pytest tests/test_enrichment_flow.py -v

# Với coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

Test suite dùng **fake providers** — mọi thứ chạy in-process, không I/O thật.

Các test bao gồm:

| File | Nội dung |
|---|---|
| `test_event_bus.py` | publish, subscribe, thread-safe bridge, exception isolation |
| `test_session_manager.py` | tạo segment, patch speaker/translation, state machine enriched |
| `test_patch_updater.py` | cả hai hướng thứ tự (speaker trước / translation trước) |
| `test_enrichment_flow.py` | E2E: `sentence_finalized` → workers → `segment_enriched` |

---

## Mở rộng backend

### Thêm speaker provider mới

Tạo class implement `SpeakerProvider`:

```python
# providers/speaker/my_provider.py
from providers.speaker.base import SpeakerProvider, SpeakerResult
from core.models import AudioSpanRef

class PyannoteProvider(SpeakerProvider):
    def __init__(self):
        from pyannote.audio import Pipeline
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    def identify(self, audio_span_ref: AudioSpanRef) -> SpeakerResult:
        # xử lý audio_span_ref.audio_bytes ở đây
        ...
        return SpeakerResult(speaker_id="speaker_1", confidence=0.88, provider="pyannote")
```

Đăng ký trong `app/main.py`:

```python
def _build_speaker_provider(config: Config):
    if config.speaker_provider == "pyannote":
        from providers.speaker.my_provider import PyannoteProvider
        return PyannoteProvider()
    ...
```

### Thêm translation provider mới

```python
# providers/translation/deepl_provider.py
from providers.translation.base import TranslationProvider, TranslationResult
import deepl

class DeepLProvider(TranslationProvider):
    def __init__(self, api_key: str):
        self._client = deepl.Translator(api_key)

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        result = self._client.translate_text(text, target_lang=target_lang.upper())
        return TranslationResult(
            translated_text=str(result),
            source_lang=source_lang,
            target_lang=target_lang,
            provider="deepl",
        )
```

Đăng ký trong `providers/translation/local_provider.py:create_translation_provider()`.

### Thay event bus bằng Redis Streams

`EventBus` chỉ cần 4 method — implement và swap vào `main.py`:

```python
class RedisEventBus:
    def subscribe(self, event_type: EventType, handler: Handler) -> None: ...
    async def publish(self, event: AnyEvent) -> None: ...
    def publish_threadsafe(self, event: AnyEvent) -> None: ...
    async def run(self) -> None: ...
    async def shutdown(self) -> None: ...
```

Không cần thay đổi bất kỳ file nào khác.

---

## Nơi lưu trữ model

| Công cụ | Đường dẫn cache |
|---|---|
| Whisper / faster-whisper | `~/.cache/huggingface/hub/` |
| argostranslate | `~/.local/share/argos-translate/` |
| resemblyzer | `~/.cache/torch/` |

Để xóa và tải lại từ đầu:
```bash
rm -rf ~/.cache/huggingface/hub/models--Systran*
rm -rf ~/.local/share/argos-translate/
```

---

## Troubleshooting

**`OSError: PortAudio library not found`**

Project dùng `sounddevice` (không dùng PyAudio). Nếu gặp lỗi:

```bash
pip install --force-reinstall sounddevice
```

Trên Linux, `sounddevice` thường có sẵn wheel. Nếu vẫn lỗi, cài `portaudio19-dev`:
```bash
sudo apt install portaudio19-dev
```

**Không tìm thấy microphone**

```bash
# Kiểm tra thiết bị audio
arecord -l
# Test ghi âm 3 giây
arecord -d 3 -f cd test.wav && aplay test.wav
```

**Whisper tải model lần đầu rất lâu**

Đây là bình thường. `base` ≈ 150 MB, `small` ≈ 470 MB. Sau lần đầu được cache lại.  
Dùng `MODEL_SIZE=tiny` (≈ 75 MB) để khởi động nhanh hơn khi test.

**`argostranslate: no package for vi→en`**

```bash
python scripts/install_argos_packages.py
```

Nếu không có internet: tải thủ công tại https://www.argosopentech.com/argospm/index/ rồi:
```bash
python -c "import argostranslate.package as p; p.install_from_path('/path/to/package.argosmodel')"
```

**Speaker luôn là `speaker_unknown`**

1. Cài `resemblyzer`: `pip install resemblyzer`
2. Kiểm tra `MIN_LENGTH_OF_RECORDING` — câu quá ngắn (< 0.5s) sẽ không có đủ audio để embed
3. Thử giảm `SPEAKER_SIMILARITY_THRESHOLD` xuống `0.65`

**Phụ đề nhấp nháy liên tục**

Tăng `REALTIME_PROCESSING_PAUSE` (vd: `0.3–0.5`) hoặc chỉnh `MIN_GROW` trong `asr/stabilizer.py`.

**Câu bị cắt quá sớm giữa chừng**

Tăng `POST_SPEECH_SILENCE_DURATION` lên `1.0–1.5`. Khi nói có nhịp ngắt giữa chừng (vd: "ừm…"), hệ thống có thể nhận đó là kết thúc câu.

**CUDA out of memory**

Dùng `COMPUTE_TYPE=int8` thay vì `float16`, hoặc giảm `MODEL_SIZE`.

**Docker: không có âm thanh từ mic**

```bash
# Kiểm tra group audio trong container
docker run --rm --device /dev/snd --group-add audio realtime-subtitle arecord -l
```

---

## Câu hỏi thường gặp

**Có hỗ trợ nhiều ngôn ngữ cùng lúc không?**  
Chưa. Một session chỉ xử lý một ngôn ngữ nguồn. Bạn có thể chạy song song nhiều instance với `SESSION_ID` khác nhau nếu cần.

**Translation có dịch được tất cả cặp ngôn ngữ không?**  
argostranslate hỗ trợ nhiều cặp ngôn ngữ (xem danh sách tại [argosopentech.com](https://www.argosopentech.com)). Mỗi cặp cần tải gói riêng qua `install_argos_packages.py`.

**Speaker attribution có chính xác không?**  
Với resemblyzer: tốt trong điều kiện microphone rõ, phòng không nhiều tiếng ồn. Không đảm bảo chính xác 100% — đây là tính năng hỗ trợ, không phải identity verification. Với heuristic fallback: chỉ dựa trên mức âm lượng, độ chính xác thấp hơn.

**Có thể lưu kết quả ra file không?**  
Hiện tại chưa có tính năng built-in. Bạn có thể redirect stdout:
```bash
python -m app.main 2>/dev/null | tee session_output.txt
```
Logs (stderr) và subtitles (stdout) được tách biệt để dễ pipe.

**Có thể dùng microphone USB / external không?**  
Có. RealtimeSTT dùng `sounddevice`, mặc định dùng default input device của hệ thống. Cấu hình default mic trong system settings hoặc thêm `input_device_index` vào `asr/realtime_recorder.py`.

**Tại sao không dùng Deepgram / AssemblyAI?**  
Dự án này được thiết kế để chạy hoàn toàn local, không phụ thuộc cloud. Bạn hoàn toàn có thể thêm cloud provider bằng cách implement interface tương ứng.

---

## Dependencies chính

| Package | Vai trò | Bắt buộc |
|---|---|---|
| `faster-whisper` | Engine transcription (CTranslate2) | Có |
| `sounddevice` | Microphone capture (không cần PyAudio) | Có |
| `numpy` | Xử lý audio PCM + energy VAD | Có |
| `resemblyzer` | Voice embedding cho speaker tracking | Không (fallback heuristic) |
| `argostranslate` | Offline translation | Không (fallback noop) |
| `pytest`, `pytest-asyncio` | Test | Dev only |
