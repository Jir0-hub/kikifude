"use client";
// Kikifude 最小デモ: ブラウザ完結（WebGPU + Transformers.js）の日本語文字起こし。
// 設計上、アカウント・課金・話者分離は持たない。
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { waitForMessage, decodeTo16kMono, type WorkerMsg } from "@/lib/worker-client";

const MODELS = {
  turbo: {
    id: "onnx-community/whisper-large-v3-turbo",
    label: "標準モデル（whisper-large-v3-turbo・推奨）",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4f16" },
  },
  kotoba: {
    id: "onnx-community/kotoba-whisper-v2.2-ONNX",
    label: "日本語特化モデル（kotoba-whisper-v2.2）",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4f16" },
  },
} as const;

// 【暫定措置 / 2026-07-20】Safari（WebKit）では実行させない。
// Mac mini M2 / Safari 26.3.1 の実測では、tiny（39M・fp16）で RTF 1.018・
// warmupMs 13,310 と、同じ機体の Chrome 145 での turbo（809M）RTF 0.137 を
// 大きく下回った。出力も10秒以降が幻覚ループで日本語として実用外。さらに
// 完走後にアプリを切り替えて戻るとタブごとクラッシュする。
// Safari は WebGPU アダプタを取得できてしまうため、既存の WebGPU 判定では
// 素通りして「実行できるが途中で無言で死ぬ」最悪の体験になる。それを止める
// のが目的。原因は未特定。
//
// 当初は WebGPU の shader-f16 非対応を疑い、capability 検出（adapter.features）
// への置き換えを検討していたが、この方針は行き止まり。Mac mini M2 / Safari
// 26.3.1 の実測（2026-07-20）で shader-f16 は YES、maxBufferSize も Chrome と
// ほぼ同値（2147483644 対 2147483648）だった。**shader-f16 の有無では Safari を
// 判別できない**ため、判定は UA ベースのまま維持する。差分の詳細と次の仮説は
// SPIKE.md「fp16 仮説の検証手順」を参照。
function isWebKitEngine(): boolean {
  // 主判定。navigator.vendor はエンジン由来の値で、WebKit ビルドだけが
  // "Apple Computer, Inc." を返す。macOS の Chrome は "Google Inc." なので
  // 誤ってブロックしない（macOS Chrome では turbo が完走している）。
  // iOS の Chrome / Firefox / Edge は中身が WebKit なのでこちらも true になる。
  if (navigator.vendor === "Apple Computer, Inc.") return true;

  const ua = navigator.userAgent;
  // 補助判定1: UA に iOS 系の文字列が残っている場合。
  if (/CriOS|FxiOS|EdgiOS|iPhone|iPad|iPod/.test(ua)) return true;
  // 補助判定2: iPadOS 13+ の「デスクトップ用サイトを表示」では UA から iPad が
  // 消えて Macintosh を名乗るため、補助判定1 では拾えない。この状態の iPad は
  // maxTouchPoints が 5 を返す一方、実機の Mac はタッチ非対応で 0 のまま
  // なので、両者を区別できる。
  return /Macintosh/.test(ua) && navigator.maxTouchPoints > 1;
}

type ModelKey = keyof typeof MODELS;
type Phase = "idle" | "loading" | "decoding" | "transcribing" | "done" | "error";

interface Chunk {
  timestamp: [number, number | null];
  text: string;
}

function formatTime(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return "--:--";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// 録音 Blob の MIME からファイル拡張子を導出する。MediaRecorder の mimeType は
// 環境依存で "audio/webm;codecs=opus"（Chrome）や "audio/mp4"（Safari 系）に
// なりうるため、パラメータ部を落としたコンテナ種別で判定する。判定不能時は
// 最も一般的な .webm にフォールバックする。
function extForMime(mime: string): string {
  const container = mime.split(";", 1)[0].trim().toLowerCase();
  switch (container) {
    case "audio/webm":
      return "webm";
    case "audio/mp4":
    case "audio/x-m4a":
      return "mp4";
    case "audio/ogg":
      return "ogg";
    case "audio/wav":
    case "audio/wave":
    case "audio/x-wav":
      return "wav";
    default:
      return "webm";
  }
}

export default function Home() {
  const workerRef = useRef<Worker | null>(null);
  const loadedModelRef = useRef<string | null>(null);
  const fileProgressRef = useRef(new Map<string, { loaded: number; total: number }>());
  const busyRef = useRef(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedRef = useRef<BlobPart[]>([]);
  // 非制御の <input type="file"> は state だけではブラウザ保持のファイル名表示を
  // 消せないため、クリア時に value を空へ戻す用の ref を持つ。
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [webgpuOk, setWebgpuOk] = useState<boolean | null>(null);
  const [isWebKit, setIsWebKit] = useState(false);
  const [modelKey, setModelKey] = useState<ModelKey>("turbo");
  const [source, setSource] = useState<{
    blob: Blob;
    name: string;
    kind: "file" | "recording";
  } | null>(null);
  const [recording, setRecording] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [statusText, setStatusText] = useState("");
  const [dlPercent, setDlPercent] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState<{
    text: string;
    chunks: Chunk[] | null;
    processMs: number;
    audioSec: number;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // WebKit はここで確定させ、アダプタの取得自体を行わない（取得できても
    // 実行させないため）。同期的に判定するので追加の再レンダリングは1回で済む。
    if (isWebKitEngine()) {
      setIsWebKit(true);
      return;
    }
    (async () => {
      if (!("gpu" in navigator)) {
        setWebgpuOk(false);
        return;
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!cancelled) setWebgpuOk(adapter !== null);
      } catch {
        if (!cancelled) setWebgpuOk(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function getWorker(): Worker {
    if (!workerRef.current) {
      const worker = new Worker(new URL("../lib/asr-worker.ts", import.meta.url), {
        type: "module",
      });
      worker.addEventListener("message", (e: MessageEvent<WorkerMsg>) => {
        if (e.data.type !== "progress") return;
        const map = fileProgressRef.current;
        map.set(String(e.data.file), {
          loaded: Number(e.data.loaded ?? 0),
          total: Number(e.data.total ?? 0),
        });
        let loaded = 0;
        let total = 0;
        for (const v of map.values()) {
          loaded += v.loaded;
          total += v.total;
        }
        if (total > 0) {
          setDlPercent(Math.min(100, (loaded / total) * 100));
          setStatusText(
            `モデルを読み込み中... ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB（初回はダウンロードのため数十秒〜数分。2回目以降はキャッシュから数秒で読み込みます）`,
          );
        }
      });
      workerRef.current = worker;
    }
    return workerRef.current;
  }

  async function transcribe() {
    if (busyRef.current || !source) return;
    busyRef.current = true;
    setErrorMsg("");
    setResult(null);
    setCopied(false);
    try {
      const worker = getWorker();

      if (loadedModelRef.current !== modelKey) {
        setPhase("loading");
        setStatusText("モデルを準備中...");
        setDlPercent(null);
        fileProgressRef.current.clear();
        const cfg = MODELS[modelKey];
        const loadedPromise = waitForMessage(worker, ["loaded"]);
        worker.postMessage({ type: "load", modelId: cfg.id, dtype: cfg.dtype });
        await loadedPromise;
        loadedModelRef.current = modelKey;
      }

      setPhase("decoding");
      setDlPercent(null);
      setStatusText("音声を読み込んでいます...");
      const { audio, durationSec } = await decodeTo16kMono(
        await source.blob.arrayBuffer(),
      );

      setPhase("transcribing");
      setStatusText(
        `文字起こし中...（音声 ${formatTime(durationSec)}。音声はこの端末の外に出ていません）`,
      );
      const resultPromise = waitForMessage(worker, ["result"]);
      worker.postMessage(
        {
          type: "transcribe",
          audio,
          durationSec,
          chunkLengthS: 30,
          strideLengthS: 5,
          returnTimestamps: true,
        },
        [audio.buffer],
      );
      const r = await resultPromise;
      setResult({
        text: String(r.text ?? "").trim(),
        chunks: (r.chunks as Chunk[] | null) ?? null,
        processMs: Number(r.processMs),
        audioSec: durationSec,
      });
      setPhase("done");
      setStatusText("");
    } catch (e) {
      setPhase("error");
      setStatusText("");
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      busyRef.current = false;
    }
  }

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recordedRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(recordedRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        setSource({ blob, name: "マイク録音", kind: "recording" });
        setRecording(false);
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (e) {
      setErrorMsg(
        `マイクを利用できませんでした: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  async function copyResult() {
    if (!result) return;
    await navigator.clipboard.writeText(result.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  // 利用者が別の音声を試せるよう、画面を完全な初期状態（ファイル未選択）へ
  // 戻す。結果・コピー表示・phase・エラー/状況表示に加え、選択中の音声（source）と
  // 非制御 file input のネイティブ表示もリセットする。input の value を空に
  // しないと、同じファイルを再選択したときに onChange が発火しない点にも対応する。
  function resetAll() {
    setResult(null);
    setCopied(false);
    setPhase("idle");
    setErrorMsg("");
    setStatusText("");
    setSource(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function downloadTxt() {
    if (!result) return;
    const url = URL.createObjectURL(
      new Blob([result.text], { type: "text/plain;charset=utf-8" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "kikifude.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  // 録音した音声（マイク由来）をローカルに保存する。すでにローカルにある
  // アップロードファイルは対象外。拡張子は Blob の MIME から導出する。
  function downloadRecording() {
    if (!source || source.kind !== "recording") return;
    const url = URL.createObjectURL(source.blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kikifude-recording.${extForMime(source.blob.type)}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const busy = phase === "loading" || phase === "decoding" || phase === "transcribing";

  return (
    <div className="container">
      <header>
        <h1 className="site-title">Kikifude（キキフデ）</h1>
        <p className="tagline">
          音声を、ブラウザだけで文字に。— 無料の日本語文字起こし（ベータ版デモ）
        </p>
        <p className="privacy-badge">
          🔒 音声はサーバーに送信されません — すべてお使いのブラウザ内で処理されます
        </p>
      </header>

      {isWebKit ? (
        <div className="card">
          <h2>Safari では現在ご利用いただけません</h2>
          <p>
            Safari と、iPhone・iPad のすべてのブラウザでは、文字起こしが正しく動作しないことを
            確認しています。処理がとても遅いうえ、途中から文章が壊れてしまい、
            さらに処理の途中でタブが強制終了することもあります。
          </p>
          <p>
            誤った結果をお見せしないよう、これらの環境では実行できないようにしています。
            原因は特定できていません。
          </p>
          <p>
            <strong>Chrome または Edge の最新版</strong>でお試しください（Windows・Mac
            のどちらでも動作します）。
          </p>
        </div>
      ) : webgpuOk === false ? (
        <div className="card">
          <h2>お使いの環境では実行できません</h2>
          <p>
            このデモは WebGPU に対応したブラウザが必要です。
            <strong>Chrome または Edge の最新版</strong>でお試しください。
          </p>
        </div>
      ) : (
        <>
          <section className="card">
            <h2>1. 音声を用意する</h2>
            <div className="row">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,.mp3,.wav,.m4a"
                aria-label="音声ファイルを選択"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setSource({ blob: f, name: f.name, kind: "file" });
                }}
                disabled={busy || recording}
              />
              <button
                type="button"
                className={recording ? "btn-recording" : ""}
                onClick={toggleRecording}
                disabled={busy}
              >
                {recording ? "■ 録音を停止" : "🎙 マイクで録音"}
              </button>
            </div>
            {source ? <p className="file-name">選択中: {source.name}</p> : null}
            {source?.kind === "recording" ? (
              <div className="row">
                <button
                  type="button"
                  onClick={downloadRecording}
                  aria-label="録音した音声をこの端末に保存"
                  disabled={busy}
                >
                  録音を保存
                </button>
              </div>
            ) : null}
          </section>

          <section className="card">
            <h2>2. モデルを選んで文字起こし</h2>
            <div className="row">
              <select
                value={modelKey}
                aria-label="モデルを選択"
                onChange={(e) => setModelKey(e.target.value as ModelKey)}
                disabled={busy}
              >
                {Object.entries(MODELS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="btn-primary"
                onClick={transcribe}
                disabled={busy || !source || webgpuOk !== true}
              >
                {busy ? "処理中..." : "文字起こしを開始"}
              </button>
            </div>
            {statusText ? <p className="status">{statusText}</p> : null}
            {dlPercent != null ? (
              <div
                className="progress-track"
                role="progressbar"
                aria-valuenow={Math.round(dlPercent)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div className="progress-fill" style={{ width: `${dlPercent}%` }} />
              </div>
            ) : null}
            {errorMsg ? <div className="error-box">エラー: {errorMsg}</div> : null}
          </section>

          {result ? (
            <section className="card">
              <h2>結果</h2>
              <pre className="result-text">
                {result.text || "（無音または認識できませんでした）"}
              </pre>
              <div className="row">
                <button type="button" onClick={copyResult}>
                  {copied ? "✓ コピーしました" : "コピー"}
                </button>
                <button type="button" onClick={downloadTxt}>
                  .txt をダウンロード
                </button>
                <button type="button" onClick={resetAll}>
                  クリア
                </button>
                <span className="file-name">
                  {formatTime(result.audioSec)} の音声を{" "}
                  {(result.processMs / 1000).toFixed(1)} 秒で処理
                </span>
              </div>
              {result.chunks && result.chunks.length > 0 ? (
                <details className="chunks">
                  <summary>タイムスタンプ付きで表示</summary>
                  {result.chunks.map((c, i) => (
                    <p className="chunk-line" key={i}>
                      [{formatTime(c.timestamp[0])} → {formatTime(c.timestamp[1])}]{" "}
                      {c.text}
                    </p>
                  ))}
                </details>
              ) : null}
            </section>
          ) : null}
        </>
      )}

      <footer>
        <p>
          Kikifude
          は<a href="https://github.com/Jir0-hub/kikifude">オープンソース</a>です。音声・文字起こし結果が外部に送信されないことは、ブラウザの開発者ツール（ネットワークタブ）でいつでも確認できます。外部への通信は、モデルファイルと実行ランタイム（WASM）の取得のみです。
        </p>
        <p>
          © 2026 Jir0 · <Link href="/spike">ベンチマークページ（開発者向け）</Link>
        </p>
      </footer>
    </div>
  );
}
