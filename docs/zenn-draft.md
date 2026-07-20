---
title: "日本語特化の kotoba-whisper をブラウザ（WebGPU）で動かし、whisper-large-v3-turbo と実測比較した"
emoji: "🎙️"
type: "tech"
topics: ["whisper", "webgpu", "transformersjs", "nextjs", "機械学習"]
published: false
---

<!-- 公開前チェックリスト（このコメントは削除してください）
- [ ] デモ URL を「デモ公開について」節に差し込む
- [ ] GitHub リポジトリ URL を差し込む（2箇所: 実装の要点・末尾）
- [ ] スクリーンショット（デモ画面 / ベンチ画面）を挿入するなら Zenn に画像アップロードして差し込む
- [ ] published: true に変更
-->

音声を**サーバーに一切送信せず**、ブラウザの中だけで日本語の文字起こしをしたい。議事録や顧客との通話、取材音声のように「クラウドにアップロードすること自体が NG」な音声は意外と多く、ローカル完結の文字起こしには一定の需要があります。

そこで、日本語特化 Whisper として評価の高い **kotoba-whisper** の ONNX 版を WebGPU + Transformers.js v4 でブラウザ上で動かし、**whisper-large-v3-turbo と速度・品質を実測比較**しました。kotoba-whisper の ONNX 版をブラウザで動かした事例はほぼ見当たらないため、実測値そのものに価値があると考えて記事にしています。

結論を先に書くと:

- **速度は両モデルとも実用域**。10分音声の処理が turbo で約1.2分、kotoba で約2.7分（RTX 4080 SUPER）
- ただし kotoba-whisper は**合成音声（TTS）入力で出力が大きく崩れる**ことを発見。自然な人間の発話では正常
- 品質・速度・入力頑健性を総合すると、ブラウザ用途のデフォルトは **turbo が優位**という結果

## 検証環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| ブラウザ | Chrome 150（WebGPU 有効） |
| ランタイム | @huggingface/transformers 4.2.0（Transformers.js） |
| フレームワーク | Next.js 16（static export、推論は Web Worker） |

:::message
ハイエンド GPU での計測です。普及帯 GPU・内蔵 GPU・スマートフォンでの数値は今後追試予定で、この記事の速度はあくまで「上限側の参考値」です。
:::

## kotoba-whisper とは

[kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) は Kotoba Technologies が公開している日本語特化の Whisper 蒸留モデルです。distil-whisper 方式で whisper-large-v3 から蒸留されており（デコーダ2層）、日本語 ASR ベンチマークで large-v3 に匹敵する精度と数倍の速度を謳っています。ライセンスは Apache 2.0。

ブラウザで動かすには ONNX 変換版が必要で、[onnx-community/kotoba-whisper-v2.2-ONNX](https://huggingface.co/onnx-community/kotoba-whisper-v2.2-ONNX) が存在します。ただしこれは機械的な変換リポジトリで、ブラウザでの動作報告はほぼ見つかりませんでした。今回はここを実機で確かめます。

なお v2.2 の目玉である話者分離・句読点付与は Python 側パイプラインの機能で、ONNX 版には含まれません（ブラウザで使えるのは ASR 本体のみ）。

## 実装の要点

構成はシンプルで、Next.js のページから Web Worker を起動し、Worker 内で Transformers.js の ASR パイプラインを実行します。

```ts
// worker.ts（抜粋）
import { pipeline } from "@huggingface/transformers";

const asr = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/kotoba-whisper-v2.2-ONNX",
  {
    device: "webgpu",
    // モジュール別に dtype を指定できる。エンコーダ fp16 + デコーダ int4 系が定番
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4f16" },
    progress_callback: (p) => self.postMessage({ type: "progress", ...p }),
  },
);

const output = await asr(audioFloat32Array, {
  language: "ja",
  task: "transcribe",
  chunk_length_s: 30,
  stride_length_s: 5,
  return_timestamps: true,
});
```

音声は入力ファイルを `OfflineAudioContext`（16kHz mono）の `decodeAudioData` でデコードして `Float32Array` を渡すだけです。`decodeAudioData` はコンテキストのサンプルレートに自動リサンプルしてくれるので、ffmpeg.wasm なしで mp3 / wav / m4a あたりまで処理できます。

```ts
const ctx = new OfflineAudioContext(1, 16000, 16000);
const decoded = await ctx.decodeAudioData(await file.arrayBuffer());
const audio = decoded.getChannelData(0); // 16kHz mono Float32Array
```

### つまずきポイント

Next.js 16（Turbopack）+ Transformers.js v4 で3つハマったので共有します。

1. **package.json に `"type": "module"` が必要**。`npm init -y` のままだと Turbopack が「Specified module format (CommonJs) is not matching…」でビルドエラーになります
2. **TypeScript は 5 系を使う**。素直に `npm i -D typescript` すると 7 系（ネイティブ版）が入る環境がありますが、Next.js 16 の型チェック統合がクラッシュしました（`The "id" argument must be of type string`）。`typescript@^5` で解決
3. **`navigator.gpu` の型には `@webgpu/types` が必要**。`@types/*` スコープではないため自動で読み込まれず、tsconfig の `types` 配列への明示が必要です

また計測面では、`progress_callback` のイベントが**モデルファイル取得の最後までずっと発火し続ける**ため、「ダウンロード時間」と「セッション初期化時間」をコールバックだけで正確に分離するのは難しいです（今回はロード合計と、キャッシュ済み2回目以降のロード時間≒初期化コスト、で代用しました）。初回推論時には WebGPU のシェーダーコンパイルが走るので、1秒の無音でウォームアップしてから本計測しています。

## ベンチマーク結果

テスト音声は2種類使いました。

- **合成音声 70.5秒**: ビジネス会議風の自作原稿（固有名詞・数字入り）を Windows の SAPI 音声（Microsoft Haruka）で合成。正解テキストがあるため品質評価用
- **自然音声 26.6秒**: kotoba-tech 公式リポジトリのサンプル音声（複数話者の自然な会話）

### 速度

| モデル / dtype | 初回ロード（DL 約1.4〜1.5GB 込み） | 70.5秒音声 | RTF | 10分音声換算 |
|---|---|---|---|---|
| kotoba-whisper-v2.2 fp16/q4f16 | 44.6秒 | 19.2秒 | 0.273 | 約2.7分 |
| kotoba-whisper-v2.2 fp16/q4 | — | 17.8秒 | 0.253 | 約2.5分 |
| kotoba-whisper-v2.2 fp16/fp16 | — | 18.6秒 | 0.263 | 約2.6分 |
| whisper-large-v3-turbo fp16/q4f16 | 70.6秒 | 8.7秒 | 0.123 | **約1.2分** |

- RTF（Real Time Factor）= 処理時間 ÷ 音声長。1.0 を切れば「再生より速い」
- 初回ロードは回線速度に強く依存します（今回は高速回線）。**2回目以降はブラウザキャッシュから約4秒**でロードできました
- 意外だったのは、蒸留で「速いはず」の kotoba よりも turbo の方がブラウザ実測では2倍以上速かったことです（後述の出力崩れによる無駄なトークン生成も一因とみられますが、自然音声でも turbo が速い結果でした）

### 品質 — そして kotoba の意外な弱点

**turbo は合成音声・自然音声とも、ほぼ完璧でした。** 合成音声での誤りは「第一四半期→第一市販期」のような同音異義語の変換ミス程度で、脱落・幻覚はゼロ。タイムスタンプも正確です。

一方 **kotoba-whisper は合成音声で出力が大きく崩れました**。具体的には:

- 同じフレーズの繰り返しループ(「一日、一日、一日、…」)
- 空白だけのセグメント、実在しない語の混入
- タイムスタンプの逆行(chunk 内で時刻が巻き戻る)
- 「田中さん」「進捗率80%」など固有名詞・数値の脱落

原因の切り分けとして次を試しましたが、いずれも症状は変わりませんでした。

| 仮説 | 検証 | 結果 |
|---|---|---|
| int4 量子化の劣化 | デコーダを q4f16 → q4 → **fp16** に変更 | 全て再現(量子化は無関係) |
| チャンク統合処理のバグ | 28秒に切り出して**単一チャンク**で実行 | 再現(統合処理は無関係) |
| チャンク長の不適合 | chunk_length_s を 30 → 25 に変更 | 再現 |
| 入力音声の分布 | **自然音声に差し替え** | **正常化** |

つまりモデル変換や量子化の問題ではなく、**学習分布から大きく外れた入力(ロボット的な合成音声)に対する頑健性**の問題です。kotoba-whisper は日本語の自然発話(ReazonSpeech = 日本のTV音声)で蒸留されており、その範囲では公称どおりの性能が出ると考えるのが妥当です。自然音声のサンプルでは kotoba も破綻なく動作しました(ただし同じ音声で比較すると、複数話者の重なり部分の拾い方は turbo が上でした。v2.2 本来の強みである話者分離パイプラインが ONNX 版に無いことも効いていそうです)。

これは kotoba-whisper の欠陥というより、**蒸留モデル一般のトレードオフ**(パラメータ削減と引き換えに分布外への頑健性が落ちる)がブラウザという「どんな音質の入力が来るか分からない」環境で顕在化した、と解釈しています。会議室の遠いマイク、圧縮の強い通話録音など、実運用の音声は案外「分布外」を含みます。不特定多数に配るブラウザツールのデフォルトモデルとしては、頑健性は精度と同じくらい重要でした。

## 結論

- **kotoba-whisper ONNX はブラウザ + WebGPU で動く**。速度は実用域(10分音声 ≒ 2.7分)
- ただし入力頑健性の懸念から、ブラウザツールのデフォルトには **whisper-large-v3-turbo(q4f16)を推奨**。速度(10分音声 ≒ 1.2分)・品質・頑健性の全てで優位だった
- kotoba-whisper は自然な日本語発話に限れば有力な選択肢。用途が限定できるなら「切替オプション」として併載する価値がある

## デモ公開について

この検証をベースに、**ブラウザ完結・無料の日本語文字起こしツール「Kikifude(キキフデ)」**を公開準備中です(モデルは turbo / kotoba を切替可能。音声はサーバーに送信されず、通信はモデル取得のみ — DevTools のネットワークタブで確認できます)。

<!-- TODO: デモ URL（https://kikifude.com 予定）/ GitHub リポジトリ URL をここに差し込む -->

コードは MIT ライセンスで全て公開します。計測に使ったベンチマークページ(モデル・dtype・チャンク長を URL パラメータで切替)も同梱しているので、手元の GPU で追試してもらえたら嬉しいです。

---

筆者: Jir0([@FolioInk_dev](https://x.com/FolioInk_dev))。ブラウザ完結ツールやサブスク型ブログプラットフォーム [FolioInk](https://folioink.com) などを個人開発しています。
