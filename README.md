# Kikifude（キキフデ／聞き筆）

**音声を、ブラウザだけで文字に。** — ブラウザ完結・無料の日本語文字起こしツール（ベータ）

**▶ https://kikifude.com**

音声ファイルをサーバーにアップロードせず、すべてお使いのブラウザ内（WebGPU）で文字起こしします。議事録・顧客通話・取材音声など、**外部に出せない機密音声**のための文字起こしツールです。

> English summary at the bottom.

## 特徴

- **音声はサーバーに送信されません** — 推論はすべてブラウザ内で完結します。外部への通信は、初回のモデルファイル取得（Hugging Face CDN）と実行ランタイムの取得（onnxruntime-web の WASM、jsDelivr CDN）だけです。**音声データと文字起こし結果は一切送信されません**。開発者ツールのネットワークタブでいつでも検証できます
- **無料・アカウント不要** — コア機能は恒久無料です
- **モデルを選べます** — 標準は `whisper-large-v3-turbo`、切替で日本語特化の `kotoba-whisper-v2.2` を利用できます（いずれも ONNX / WebGPU 実行）
- ファイルアップロード（mp3 / wav / m4a 等）とマイク録音に対応
- タイムスタンプ表示、コピー、.txt ダウンロード

## 動作環境

**Chrome または Edge の最新版**（Windows / macOS）を推奨します。WebGPU 対応ブラウザが必要です。初回はモデルのダウンロード（約 1.4GB）が発生します。2回目以降はブラウザキャッシュから読み込みます。

**Safari および iPhone / iPad は現在ご利用いただけません。** Safari は WebGPU に対応していますが、実測で速度・品質ともに実用に耐えず、処理中にタブが強制終了する事象も確認しています。検証では、このデモで使う2モデルよりはるかに小さい `whisper-tiny`（デモ標準の `whisper-large-v3-turbo` の約20分の1の規模。動作確認用で、デモの選択肢には含まれません）ですら、Mac mini M2 / Safari 26.3.1 で音声の長さと同じだけ処理時間がかかり（RTF 1.018。1.0 で等速、小さいほど速い）、出力も途中から破綻しました。デモのモデルはこれより大きいため、さらに実用外になります。誤った結果を出さないよう、UI 側で実行前にブロックしています。原因は調査中です。同じ Mac でも Chrome では正常に動作します。

WebGPU 非対応の環境では現在ご利用いただけません。

ソースコード: https://github.com/Jir0-hub/kikifude

## 技術構成

- Next.js（App Router / static export）+ TypeScript
- [@huggingface/transformers](https://github.com/huggingface/transformers.js)（Transformers.js v4）+ WebGPU、推論は Web Worker で実行
- モデル: [onnx-community/whisper-large-v3-turbo](https://huggingface.co/onnx-community/whisper-large-v3-turbo) / [onnx-community/kotoba-whisper-v2.2-ONNX](https://huggingface.co/onnx-community/kotoba-whisper-v2.2-ONNX)（encoder fp16 / decoder q4f16）
- バックエンドなし・データベースなし・トラッキングなし（アクセス解析は導入時に README に明記します）

### 外部への通信（すべて DevTools で検証できます）

| 通信先 | 内容 | タイミング |
|---|---|---|
| Hugging Face CDN | ONNX モデル・トークナイザ等の設定ファイル | 初回のみ（以降はブラウザキャッシュ） |
| jsDelivr CDN | onnxruntime-web の WASM ランタイム | 初回のみ |

**音声データ・文字起こし結果を送信する通信は存在しません。**

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

**Kikifude** is a free, in-browser Japanese speech-to-text tool. Audio never leaves your device: inference runs entirely in the browser via WebGPU (Transformers.js v4 + ONNX Whisper models in a Web Worker). The only outbound traffic is fetching the model files (Hugging Face CDN, first run only) and the ONNX Runtime WASM binaries (jsDelivr CDN) — **no audio and no transcript is ever uploaded**. Verifiable in DevTools.

- Default model: `whisper-large-v3-turbo`, switchable to the Japanese-specialized `kotoba-whisper-v2.2`
- File upload + microphone recording, timestamps, copy / .txt download
- No backend, no account, no tracking
- Requires a WebGPU-capable browser: **latest Chrome or Edge** (Windows / macOS). Safari and all iPhone / iPad browsers are currently blocked — WebGPU is available there, but measured speed and output quality are unusable and the tab can be killed mid-run (root cause still under investigation). Chrome on the same Mac works fine.

Benchmarks and reproduction steps: see [SPIKE.md](./SPIKE.md). License: MIT.
