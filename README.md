# Kikifude（キキフデ／聞き筆）

**音声を、ブラウザだけで文字に。** — ブラウザ完結・無料の日本語文字起こしツール（ベータ）

**▶ https://kikifude.com**

音声ファイルをサーバーにアップロードせず、すべてお使いのブラウザ内（WebGPU）で文字起こしします。議事録・顧客通話・取材音声など、**外部に出せない機密音声**のための文字起こしツールです。

> English summary at the bottom.

## 特徴

- **音声はサーバーに送信されません** — 推論はすべてブラウザ内で完結します。外部通信はモデルファイルの取得（Hugging Face CDN、初回のみ）だけで、開発者ツールのネットワークタブでいつでも検証できます
- **無料・アカウント不要** — コア機能は恒久無料です
- **モデルを選べます** — 標準は `whisper-large-v3-turbo`、切替で日本語特化の `kotoba-whisper-v2.2` を利用できます（いずれも ONNX / WebGPU 実行）
- ファイルアップロード（mp3 / wav / m4a 等）とマイク録音に対応
- タイムスタンプ表示、コピー、.txt ダウンロード

## 動作環境

WebGPU 対応ブラウザが必要です（Chrome / Edge 最新版、Safari 26 以降など）。初回はモデルのダウンロード（約 1.4GB）が発生します。2回目以降はブラウザキャッシュから読み込みます。

WebGPU 非対応環境向けの軽量モデルフォールバックは開発中です。

## 技術構成

- Next.js（App Router / static export）+ TypeScript
- [@huggingface/transformers](https://github.com/huggingface/transformers.js)（Transformers.js v4）+ WebGPU、推論は Web Worker で実行
- モデル: [onnx-community/whisper-large-v3-turbo](https://huggingface.co/onnx-community/whisper-large-v3-turbo) / [onnx-community/kotoba-whisper-v2.2-ONNX](https://huggingface.co/onnx-community/kotoba-whisper-v2.2-ONNX)（encoder fp16 / decoder q4f16）
- バックエンドなし・データベースなし・トラッキングなし（アクセス解析は導入時に README に明記します）

## ベンチマーク（実測）

RTX 4080 SUPER / Chrome 150 での実測（詳細な条件と再現手順は [SPIKE.md](./SPIKE.md)、計測ページは `/spike`）:

| モデル | 70.5秒音声の処理 | RTF | 10分音声換算 |
|---|---|---|---|
| whisper-large-v3-turbo (q4f16) | 8.7秒 | 0.123 | 約1.2分 |
| kotoba-whisper-v2.2 (q4f16) | 19.2秒 | 0.273 | 約2.7分 |

- RTF = 処理時間 ÷ 音声長（小さいほど速い）
- kotoba-whisper は合成音声（TTS）など学習分布から離れた入力で出力が崩れることを確認しています（自然な人間の発話では正常）。このため標準モデルは turbo にしています

## 開発

```bash
npm install
npm run dev    # http://localhost:3000
npm run build  # static export → out/
```

`/spike` はモデル・dtype・チャンク長をパラメータで切り替えられるベンチマークページです（使い方は [SPIKE.md](./SPIKE.md)）。

## ライセンス

[MIT](./LICENSE) © 2026 [Jir0](https://jir0.com)

利用モデルのライセンス: kotoba-whisper-v2.2 は Apache 2.0。whisper-large-v3-turbo は配布元（Hugging Face モデルカード）を参照してください。

---

## English

**Kikifude** is a free, in-browser Japanese speech-to-text tool. Audio never leaves your device: inference runs entirely in the browser via WebGPU (Transformers.js v4 + ONNX Whisper models in a Web Worker). The only network traffic is the one-time model download from the Hugging Face CDN — verifiable in DevTools.

- Default model: `whisper-large-v3-turbo`, switchable to the Japanese-specialized `kotoba-whisper-v2.2`
- File upload + microphone recording, timestamps, copy / .txt download
- No backend, no account, no tracking
- Requires a WebGPU-capable browser (latest Chrome / Edge, Safari 26+)

Benchmarks and reproduction steps: see [SPIKE.md](./SPIKE.md). License: MIT.
