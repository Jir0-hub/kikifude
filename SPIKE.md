# Kikifude G0 技術スパイク

kotoba-whisper の ONNX 版がブラウザ（WebGPU + Transformers.js v4）で実用速度・実用品質で動くかを検証するスパイク。コードは使い捨て前提（charter §4）。

## 実行手順

```bash
npm install
npm run dev
# → http://localhost:3000/spike を Chrome (WebGPU 対応) で開く
```

- 画面のモデル選択 + 「Run benchmark」で手動実行
- 自動実行: `/spike?autorun=<modelKey>` （`kotoba` / `kotoba-q4` / `kotoba-fp16` / `kotoba-fp32` / `turbo`）
- 追加パラメータ: `chunk`（chunk_length_s、既定30）/ `stride`（既定5）/ `ts=0`（タイムスタンプ無効）/ `maxSec=N`（音声先頭N秒のみ）/ `src=/sample-natural.mp3`（音声差替え）
- 結果は画面表示 + DevTools コンソールに `SPIKE_RESULT {json}` の1行で出力

## テスト音声

- `public/spike-audio.wav`（70.5秒）: `spike-assets/script.txt` の原稿を SAPI（Microsoft Haruka）で合成。**ロボット音声のため kotoba-whisper には分布外**（下記の結果参照）。再生成:

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Haruka Desktop"); $s.Rate = 0
$s.SetOutputToWaveFile("spike-assets\audio-60s.wav")
$s.Speak((Get-Content -Raw -Encoding UTF8 "spike-assets\script.txt"))
$s.Dispose()
Copy-Item spike-assets\audio-60s.wav public\spike-audio.wav
```

- `public/sample-natural.mp3`（26.6秒、自然会話・複数話者）: kotoba-tech 公式サンプル。取得:

```powershell
Invoke-WebRequest "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2/resolve/main/sample_audio/sample_diarization_japanese.mp3" -OutFile public\sample-natural.mp3
```

## iPhone 実機テスト手順（未実施・ユーザー依頼事項）

WebGPU は secure context 必須のため、LAN 越しの HTTP では `navigator.gpu` が取れない。HTTPS の dev サーバーで配信する:

```bash
npm run dev -- --experimental-https
# PC とiPhone を同一 Wi-Fi に接続し、iPhone の Safari で
# https://<PCのLAN IP>:3000/spike?autorun=kotoba を開く（証明書警告は許可）
```

- iOS Safari の WebGPU 対応状況はページ上部の表示で確認できる（非対応なら赤字表示）
- モデル DL 約1.4GB が発生する点に注意（Wi-Fi 必須）

## 計測結果（2026-07-20、RTX 4080 SUPER / Chrome 150 / 高速回線）

| 条件 | ロード合計(コールド) | 70.5秒TTS音声 | RTF | 10分換算 | 品質 |
|------|------|------|------|------|------|
| kotoba q4f16 | 44.6s (1.44GB) | 19.2s | 0.273 | 2.7分 | TTS音声で崩壊 / 自然音声は正常 |
| kotoba q4 | (差分DL 11.4s) | 17.8s | 0.253 | 2.5分 | 同上 |
| kotoba fp16 | (差分DL 10.5s) | 18.6s | 0.263 | 2.6分 | 同上 → 量子化は無関係 |
| turbo q4f16 | 70.6s (1.47GB) | 8.7s | 0.123 | 1.2分 | ほぼ完璧（誤りは同音異義のみ） |

- ウォームキャッシュ時のロードは約3.5〜4秒（≒セッション初期化コスト）。warmup（シェーダコンパイル）0.6〜1.5秒
- 自然音声（26.6秒サンプル）: kotoba RTF 0.300 / turbo RTF 0.084。文字起こしの網羅性・正確性ともに turbo が優位
- kotoba の合成音声での品質崩壊は dtype（q4f16/q4/fp16）・単一チャンク・チャンク長25 のいずれでも再現し、自然音声への差し替えでのみ解消 → 分布外入力への頑健性の問題と判断。この結果を受けデフォルトモデルは turbo を採用
