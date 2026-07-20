# Kikifude G0 技術スパイク

kotoba-whisper の ONNX 版がブラウザ（WebGPU + Transformers.js v4）で実用速度・実用品質で動くかを検証するスパイク。コードは使い捨て前提（charter §4）。

## 実行手順

```bash
npm install
npm run dev
# → http://localhost:3000/spike を Chrome (WebGPU 対応) で開く
```

- 画面のモデル選択 + 「Run benchmark」で手動実行
- 自動実行: `/spike?autorun=<modelKey>` （`tiny` / `tiny-q8` / `tiny-fp32` / `base` / `small` / `small-q8` / `small-fp32` / `kotoba` / `kotoba-q4` / `kotoba-fp16` / `kotoba-fp32` / `turbo`）
- 追加パラメータ: `chunk`（chunk_length_s、既定30）/ `stride`（既定5）/ `ts=0`（タイムスタンプ無効）/ `maxSec=N`（音声先頭N秒のみ）/ `src=/sample-natural.mp3`（音声差替え）
- 結果は画面表示 + DevTools コンソールに `SPIKE_RESULT {json}` の1行で出力
- ページ上部にアダプタの capability（`shader-f16` の有無 / `features` / 主要 `limits`）を表示する。
  同じ値は `SPIKE_RESULT` の `gpuCaps` にも含まれる。**実行しなくても表示されるので、
  ブラウザ間の比較だけなら `/spike` を開いてスクリーンショットを撮るだけでよい**

## モデルキーと想定ダウンロードサイズ

`autorun` に指定できるキーの一覧。サイズは指定 dtype で実際に取得される ONNX ファイル
（`encoder_model_<dtype>.onnx` + `decoder_model_merged_<dtype>.onnx`）の合計値。

| キー | モデル ID | dtype (encoder / decoder) | 想定DLサイズ |
|------|-----------|---------------------------|-------------|
| `tiny` | `onnx-community/whisper-tiny` | fp16 / fp16 | 約76MB |
| `tiny-q8` | `onnx-community/whisper-tiny` | q8 / q8 | 約41MB |
| `tiny-fp32` | `onnx-community/whisper-tiny` | fp32 / fp32 | 約152MB |
| `base` | `onnx-community/whisper-base` | fp16 / fp16 | 約146MB |
| `small` | `onnx-community/whisper-small` | fp16 / fp16 | 約485MB |
| `small-q8` | `onnx-community/whisper-small` | q8 / q8 | 約249MB |
| `small-fp32` | `onnx-community/whisper-small` | fp32 / fp32 | 約968MB |
| `kotoba` | `onnx-community/kotoba-whisper-v2.2-ONNX` | fp16 / q4f16 | 約1.44GB（実測） |
| `turbo` | `onnx-community/whisper-large-v3-turbo` | fp16 / q4f16 | 約1.47GB（実測） |

`kotoba-q4` / `kotoba-fp16` / `kotoba-fp32` は kotoba の dtype 比較用。

**軽量3モデルが turbo と別の dtype を使う理由**: `onnx-community/whisper-{tiny,base,small}`
には **q4f16 のファイルが存在しない**（`fp32` / `fp16` / `q4` / `int8` / `uint8` / `bnb4` のみ）。
turbo と同じ `q4f16` を指定するとロード時に 404 になる。加えて tiny/base はデコーダのサイズを
トークン埋め込みが占め、q4 はそこを fp32 のまま残すため **q4 の方が fp16 より大きい**
（tiny: q4 86.7MB / fp16 59.6MB）。fp16 に揃えることでサイズが単調に増える階段になり、
比較しやすい。

## fp16 仮説の検証手順（Safari / WebKit）

Mac mini M2 / Safari 26.3.1 で `tiny`（39M・fp16）が RTF 1.018 / warmupMs 13,310 だったのに対し、
同一機体の Chrome 145 では `turbo`（809M・約20倍）が RTF 0.137 / warmupMs 2,829 だった。
**20分の1のモデルが7倍以上遅い**のは容量の問題では説明がつかない。

当初は WebGPU の `shader-f16` が使えず fp16 の重みが変換を挟んでいる、という仮説を立てていた。

### 実測結果: `shader-f16` 仮説は否定された（2026-07-20）

`/spike` の capability 表示を両機で突き合わせた結果、**Safari 側も `shader-f16` を持っていた**。

| 項目 | Mac mini M2 / Safari 26.3.1 | RTX 4080 SUPER / Chrome 150 |
|------|------|------|
| `shader-f16` | **YES** | YES |
| `maxBufferSize` | 2,147,483,644 | 2,147,483,648 |
| `maxStorageBufferBindingSize` | 2,147,483,644 | 2,147,483,644 |

`maxBufferSize` の差は 4 バイトで、メモリ上限の差でもない。**`shader-f16` の有無・バッファ上限の
どちらも遅さの説明にならない。**この結論を受け、アプリ本体（`app/page.tsx`）の Safari 判定を
`adapter.features.has("shader-f16")` による capability 検出へ置き換える案は破棄し、UA ベースの
判定を維持している。

### 手順を残す理由

上記のとおり `shader-f16` 仮説は死んだが、**dtype の A/B 自体はまだ有効**。「f16 がサポート
されているか」と「f16 が速く実行されるか」は別の問題で、後者はサポートの有無を見ても分からない。
`tiny` / `tiny-q8` / `tiny-fp32` の比較は、**`shader-f16` 以外の要因で dtype が効いている可能性**を
潰す（あるいは拾う）ために引き続き実行する価値がある。以下の手順はそのまま使える。

**モデルと音声を固定したまま dtype から f16 を外す**ことで、これを検証する。

### なぜ `small` ではなく `tiny` で比較するのか（重要）

**同じ Mac mini の Safari では `small`（485MB）が推論中に落ちている**（読み込みは完了、
推論の途中でタブが死ぬ）。したがって `small` では基準値が取れず、その2倍ある
`small-fp32`（968MB）はさらに確実に落ちる。**この機体で比較を組むと原理的に成立しない。**

**同じ Safari で完走実績があるのは `tiny`（76MB / RTF 1.018 / warmupMs 13,310）だけ**なので、
これを主系列にする。既に基準値が取れているうえ、`tiny-fp32` でも152MB と `small` の
1/3 以下に収まり、完走が見込める条件のまま同じ結論が得られる。

### 手順（Safari 実機）

**`/spike` は Safari でも実行できる。** アプリ本体（`/`）は 2026-07-20 から Safari を
実行前にブロックしているが、`/spike` はこの調査そのものに必要なため意図的に対象外に
してある。「本体は止まるのに `/spike` は動く」のは仕様。

| 実行 | dtype | 想定DL | 見るところ |
|------|-------|--------|-----------|
| 1. `/spike` を開くだけ | — | 0 | capability の記録（実施済み・上表を参照） |
| 2. `/spike?autorun=tiny` | fp16 / fp16 | 約76MB | 基準値（既知: RTF 1.018 / warmup 13,310ms） |
| 3. `/spike?autorun=tiny-q8` | q8 / q8 | 約41MB | f16 なし・最小サイズ |
| 4. `/spike?autorun=tiny-fp32` | fp32 / fp32 | 約152MB | f16 なし・最大サイズ |

`tiny-fp32` はダウンロードが `tiny` の2倍あるにもかかわらず速い（特に `warmupMs` が縮む）、
という結果が出れば、遅さの原因はモデルサイズではなく fp16 の取り扱いだと確定できる。

### `small-q8` / `small-fp32` の位置づけ

**Chrome 側で dtype 差を対照するための系列**であり、Safari 機では実行しない（上記のとおり
`small` 自体が落ちるため）。この負荷を問題なくこなすブラウザで同じ dtype 差がどう出るかを
デスクトップ Chrome で測っておくと、Safari の `tiny-*` の数値を単独ではなく
既知の対照と突き合わせて読める。

### dtype 追加時の注意

`q8` / `fp32` は tiny / small とも HF のファイルツリーで実在を確認済み
（`q8` → `*_quantized.onnx`、`fp32` → 無サフィックス）。**存在しない dtype を指定すると
404 になり「メモリ不足で落ちた」と誤読される**ため、キーを追加する際は必ず
ファイルの実在を確認すること（tiny/base/small に `q4f16` が無いのは既知）。

## 次の仮説: `subgroups` 非対応（**未検証**）

> **これは未検証の仮説である。**`shader-f16` を同じように「有力」と書いて否定された直後なので、
> 裏付けが取れるまで README・UI を含むユーザー向けの記述には一切書かないこと。

### 確認できた事実

**1. features の差分は、計算関連では `subgroups` だけ**

両機の `features` を突き合わせると、Chrome にあって Safari にないのは以下:

| feature | 用途 |
|---------|------|
| `subgroups` | **GPU 計算**（ワープ内スレッド間のデータ交換） |
| `clip-distances` | グラフィックス描画 |
| `dual-source-blending` | グラフィックス描画 |
| `primitive-index` | グラフィックス描画 |
| `texture-component-swizzle` | グラフィックス描画 |
| `texture-formats-tier2` | テクスチャ形式 |

下5つはいずれも描画用で ML の行列演算には無関係。**計算に効きうる差分は `subgroups` のみ。**

**2. onnxruntime-web に `subgroups` の分岐が実在する**（`onnxruntime-web@1.26.0-dev.20260416`）

`node_modules/onnxruntime-web/dist/ort.mjs` に、対応時のみ WGSL の `enable` ディレクティブを
出す分岐がある:

```js
const extensionsInfo = [
  { feature: "shader-f16", extension: "f16" },
  { feature: "subgroups", extension: "subgroups" }
];
extensionsInfo.forEach((info) => {
  if (device.features.has(info.feature)) {
    enableDirectives.push(`enable ${info.extension};`);
  }
});
```

デバイス生成時にも `requireFeatureIfAvailable("shader-f16")` と並んで
`requireFeatureIfAvailable("subgroups")` を要求している（**あれば使う**、無ければ黙って続行）。

**3. カーネル側に subgroup 版と非 subgroup 版の両方が存在する**

`ort-wasm-simd-threaded.jspi.wasm` / `.asyncify.wasm`（ネイティブ WebGPU EP）の WGSL 文字列に
`subgroupShuffle` / `subgroupBroadcast` / `subgroupMax` / `subgroup_invocation_id` が含まれる。
同一カーネルの2系統が隣接して埋め込まれている:

```
sum += fetchVTile(15, i, val) * qk_4[3];        ← 非 subgroup 版
sum += subgroupShuffle(val, 15) * qk_4[3];      ← subgroup 版
```

`qk_` / `fetchVTile` という名前から **attention の計算カーネル**と読める。
なお `ort-wasm-simd-threaded.jsep.wasm`（JSEP 版）には subgroup 文字列が**1つも無い**。

### 推測（裏付けなし）

- 上記の非 subgroup 版フォールバックが subgroup 版より十分に遅ければ、Safari の RTF 1.018 /
  warmup 13,310ms を説明しうる。**ただし両系統の性能差は測っていない**
- **実行時にどの wasm バリアントがロードされているか未確認。**Transformers.js 経由 + jsDelivr
  配信のため、jspi / asyncify / jsep のどれが選ばれるかは環境依存。JSEP 版なら subgroup 版
  カーネルはそもそも存在せず、この仮説は Safari 固有の説明にならない
- 出力破綻（幻覚ループ）とタブ強制終了は、速度低下とは別原因の可能性がある。
  `subgroups` は速度は説明しても**品質の破綻を説明しない**

### 検証するなら

1. Safari 実行時に実際にロードされている `.wasm` のファイル名を DevTools のネットワークタブで確認する
2. Chrome を `--disable-dawn-features=use_subgroups` 等で subgroups 無効化して起動し、
   同じ機体・同じモデルで RTF と warmupMs が Safari 側に近づくかを見る（**対照が同一機体で取れる**
   ため、これが最も筋が良い）

## メモリ上限の切り分け

メモリ上限の切り分けは `tiny` → `base` → `small` → `turbo` の順に実行し、
**最初にクラッシュしたキー**を記録する。実行中のモデル名とサイズは画面上部に
黄色の帯で表示されるので、クラッシュ前にスクリーンショットを撮っておくとよい
（iOS のメモリ不足によるタブ強制終了では例外もコンソールログも残らない）。

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

## スマホ実機テスト手順（iPhone / iPad）

WebGPU は secure context 必須のため、LAN 越しの HTTP では `navigator.gpu` が取れない。HTTPS の dev サーバーで配信する。**次の4つはいずれも必須で、1つでも欠けると失敗する。**

### 1. dev サーバーを LAN 公開で起動

```bash
npm run dev -- --experimental-https -H 0.0.0.0
```

`-H 0.0.0.0` がないと loopback のみで待ち受けるため、実機からは接続できない。

### 2. `allowedDevOrigins` に自分の LAN IP を含める

Next.js は既定でクロスオリジンからの `/_next/*`（HMR 等の dev リソース）を 403 で拒否する。これに当たると **HTML は表示されるがクライアント JS が実行されず、画面が初期値のまま固まる**（ハイドレーションが起きない）。dev サーバーのログに `Blocked cross-origin request to Next.js dev resource ...` が出ていればこれが原因。

`next.config.ts` に設定済み:

```ts
allowedDevOrigins: ["192.168.*.*"],
```

`*` は1セグメントにのみマッチする。LAN が `192.168.x.x` 以外（`10.x.x.x` など）の場合はここを書き換えて dev サーバーを再起動すること。この設定は dev 専用で、`npm run build`（static export）の成果物には影響しない。

PC の LAN IP は `ipconfig`（Windows）/ `ipconfig getifaddr en0`（macOS）で確認する。

### 3. Windows ファイアウォール

- 3000/TCP の**受信**を許可する
- **ネットワークプロファイルが「プライベート」であること**を確認する。「パブリック」のままだと Private スコープの許可規則が適用されず、規則を追加しても接続できない（実際に踏んだ）

```powershell
# プロファイル確認（NetworkCategory が Private であること）
Get-NetConnectionProfile
# 受信許可（プライベートネットワーク向け）
New-NetFirewallRule -DisplayName "Next dev 3000" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow -Profile Private
```

### 4. 実機からアクセス

PC と iPhone を同一 Wi-Fi に接続し、Safari で `https://<PCのLAN IP>:3000/spike?autorun=turbo` を開く。

- `--experimental-https` の自己署名証明書は SAN に `localhost` / `127.0.0.1` / `0.0.0.0` しか持たないため、**LAN IP でのアクセスでは必ず証明書警告が出る**。「詳細」→「このWebサイトを閲覧」で承認すれば secure context として扱われ、`navigator.gpu` は取得できる
- iOS Safari の WebGPU 対応状況はページ上部の表示で確認できる（非対応なら赤字表示）
- モデル DL 約1.4GB が発生する点に注意（Wi-Fi 必須・ギガ消費に注意）

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
