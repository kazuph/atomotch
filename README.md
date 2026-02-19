# atomotch

AtomS3R + MioTTS で作る、しゃべるゴッチ風育成ペット。

## 構成

```
atomotch/
├── device/     # ESP32ファームウェア (PlatformIO / M5Unified)
│   ├── src/
│   │   ├── main.cpp              # メインアプリケーション
│   │   └── robot_voice_effects.h # DSPエフェクト
│   └── platformio.ini
└── infra/      # サーバー (Docker Compose)
    ├── docker-compose.yml  # vllm + miotts-api + stt-server + web
    ├── Dockerfile
    ├── stt_server.py       # STT (Whisper) + Gemini LLM
    ├── miotts_server/      # MioTTS音声合成
    └── .env                # GEMINI_API_KEY等
```

## サービス

| サービス | ポート | 役割 |
|----------|--------|------|
| vllm | 内部 | MioTTS用LLMバックエンド |
| miotts-api | 8001 | 日本語音声合成 (TTS) |
| stt-server | 8002 | 音声認識 (STT) + Gemini応答 |
| miotts-web | 7860 | Gradio Web UI |

## 起動

```bash
cd infra
docker compose up -d
```

## デバイス

AtomS3R (ESP32-S3) + Atomic Echo Base (ES8311)

```bash
cd device
pio run -t upload
```
