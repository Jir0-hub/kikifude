"use client";
// Kikifude 最小デモ: ブラウザ完結（WebGPU + Transformers.js）の日本語文字起こし。
// アカウント・課金・話者分離なし（charter §3）。非WebGPU環境のフォールバックは MVP で対応。
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

export default function Home() {
  const workerRef = useRef<Worker | null>(null);
  const loadedModelRef = useRef<string | null>(null);
  const fileProgressRef = useRef(new Map<string, { loaded: number; total: number }>());
  const busyRef = useRef(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedRef = useRef<BlobPart[]>([]);

  const [webgpuOk, setWebgpuOk] = useState<boolean | null>(null);
  const [modelKey, setModelKey] = useState<ModelKey>("turbo");
  const [source, setSource] = useState<{ blob: Blob; name: string } | null>(null);
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
        setSource({ blob, name: "マイク録音" });
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

      {webgpuOk === false ? (
        <div className="card">
          <h2>お使いの環境では実行できません</h2>
          <p>
            このデモは WebGPU 対応ブラウザが必要です（Chrome / Edge 最新版、Safari 26
            以降など）。WebGPU 非対応環境向けの軽量モードは現在開発中です。
          </p>
        </div>
      ) : (
        <>
          <section className="card">
            <h2>1. 音声を用意する</h2>
            <div className="row">
              <input
                type="file"
                accept="audio/*,.mp3,.wav,.m4a"
                aria-label="音声ファイルを選択"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setSource({ blob: f, name: f.name });
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
