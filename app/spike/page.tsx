"use client";
// G0 spike page: benchmarks kotoba-whisper ONNX (and fallback model) on WebGPU.
// Results are rendered on screen and logged as one-line JSON ("SPIKE_RESULT ...")
// so an automated harness can scrape them from the console.
import { useEffect, useRef, useState } from "react";
import { waitForMessage, type WorkerMsg } from "@/lib/worker-client";

const MODELS: Record<
  string,
  { id: string; dtype: Record<string, string> }
> = {
  kotoba: {
    id: "onnx-community/kotoba-whisper-v2.2-ONNX",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4f16" },
  },
  "kotoba-q4": {
    id: "onnx-community/kotoba-whisper-v2.2-ONNX",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
  },
  "kotoba-fp16": {
    id: "onnx-community/kotoba-whisper-v2.2-ONNX",
    dtype: { encoder_model: "fp16", decoder_model_merged: "fp16" },
  },
  "kotoba-fp32": {
    id: "onnx-community/kotoba-whisper-v2.2-ONNX",
    dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" },
  },
  turbo: {
    id: "onnx-community/whisper-large-v3-turbo",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4f16" },
  },
};

export default function SpikePage() {
  const workerRef = useRef<Worker | null>(null);
  const runningRef = useRef(false);
  const [gpuInfo, setGpuInfo] = useState<string>("checking...");
  const [webgpuOk, setWebgpuOk] = useState<boolean | null>(null);
  const [modelKey, setModelKey] = useState("kotoba");
  const [logLines, setLogLines] = useState<string[]>([]);
  const [progress, setProgress] = useState<string>("");
  const [resultJson, setResultJson] = useState<string>("");

  const log = (line: string) =>
    setLogLines((prev) => [...prev, `[${new Date().toISOString().slice(11, 19)}] ${line}`]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!("gpu" in navigator)) {
        setWebgpuOk(false);
        setGpuInfo("navigator.gpu is not available (WebGPU unsupported)");
        return;
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (cancelled) return;
        if (!adapter) {
          setWebgpuOk(false);
          setGpuInfo("navigator.gpu present but requestAdapter() returned null");
          return;
        }
        const info = adapter.info;
        setWebgpuOk(true);
        setGpuInfo(
          `WebGPU OK — vendor=${info?.vendor ?? "?"} architecture=${info?.architecture ?? "?"} description=${info?.description ?? "?"}`,
        );
      } catch (e) {
        setWebgpuOk(false);
        setGpuInfo(`WebGPU error: ${String(e)}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function runBenchmark(
    key: string,
    chunkLengthS = 30,
    strideLengthS = 5,
    returnTimestamps = true,
    maxSec = 0,
    src = "/spike-audio.wav",
  ) {
    if (runningRef.current) return;
    runningRef.current = true;
    setResultJson("");
    const cfg = MODELS[key];
    try {
      if (!workerRef.current) {
        workerRef.current = new Worker(
          new URL("../../lib/asr-worker.ts", import.meta.url),
          { type: "module" },
        );
        workerRef.current.addEventListener("message", (e: MessageEvent<WorkerMsg>) => {
          if (e.data.type === "progress") {
            setProgress(
              `${e.data.file}: ${Number(e.data.progress ?? 0).toFixed(1)}%`,
            );
          }
        });
        await waitForMessage(workerRef.current, ["worker-ready"]);
      }
      const worker = workerRef.current;

      log(`load start: ${cfg.id} dtype=${JSON.stringify(cfg.dtype)}`);
      const loadedPromise = waitForMessage(worker, ["loaded"]);
      worker.postMessage({ type: "load", modelId: cfg.id, dtype: cfg.dtype });
      const loaded = await loadedPromise;
      log(`load done: ${JSON.stringify(loaded.metrics)}`);

      log(`fetching + decoding ${src} ...`);
      const res = await fetch(src);
      if (!res.ok) throw new Error(`audio fetch failed: HTTP ${res.status}`);
      const raw = await res.arrayBuffer();
      // decodeAudioData resamples to the context sample rate (16 kHz, mono take).
      const ctx = new OfflineAudioContext(1, 16000, 16000);
      const decoded = await ctx.decodeAudioData(raw);
      let audio = decoded.getChannelData(0);
      let durationSec = decoded.duration;
      if (maxSec > 0 && durationSec > maxSec) {
        audio = audio.slice(0, Math.floor(maxSec * 16000));
        durationSec = maxSec;
      }
      log(`audio decoded: ${durationSec.toFixed(1)}s @16kHz`);

      log(
        `transcribe start: chunk=${chunkLengthS} stride=${strideLengthS} ts=${returnTimestamps} audio=${durationSec.toFixed(1)}s`,
      );
      const resultPromise = waitForMessage(worker, ["result"]);
      worker.postMessage(
        { type: "transcribe", audio, durationSec, chunkLengthS, strideLengthS, returnTimestamps },
        [audio.buffer],
      );
      const result = await resultPromise;
      log(`transcribe done: ${result.processMs}ms RTF=${Number(result.rtf).toFixed(3)}`);

      const payload = {
        modelKey: key,
        modelId: cfg.id,
        dtype: cfg.dtype,
        chunkLengthS,
        strideLengthS,
        returnTimestamps,
        maxSec,
        src,
        gpu: gpuInfo,
        userAgent: navigator.userAgent,
        load: loaded.metrics,
        audioSec: Math.round(durationSec * 10) / 10,
        processMs: result.processMs,
        rtf: result.rtf,
        projected10minSec: result.projected10minSec,
        text: result.text,
        chunks: result.chunks,
      };
      setResultJson(JSON.stringify(payload, null, 2));
      console.log("SPIKE_RESULT " + JSON.stringify(payload));
    } catch (e) {
      const msg = `SPIKE_ERROR ${e instanceof Error ? e.message : String(e)}`;
      log(msg);
      console.log(msg);
    } finally {
      runningRef.current = false;
    }
  }

  // Autorun for the automated harness: /spike?autorun=kotoba
  useEffect(() => {
    if (webgpuOk !== true) return;
    const params = new URLSearchParams(window.location.search);
    const key = params.get("autorun");
    const chunk = Number(params.get("chunk") ?? 30);
    const stride = Number(params.get("stride") ?? 5);
    const ts = params.get("ts") !== "0";
    const maxSec = Number(params.get("maxSec") ?? 0);
    const src = params.get("src") ?? "/spike-audio.wav";
    if (key && MODELS[key]) void runBenchmark(key, chunk, stride, ts, maxSec, src);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [webgpuOk]);

  return (
    <main>
      <h1>Kikifude G0 spike</h1>
      <p data-testid="gpu-info">{gpuInfo}</p>
      {webgpuOk === false ? (
        <p style={{ color: "red" }}>
          この環境は WebGPU に対応していません。スパイクは WebGPU 環境でのみ実行できます。
        </p>
      ) : (
        <>
          <label>
            model:{" "}
            <select value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
              {Object.keys(MODELS).map((k) => (
                <option key={k} value={k}>
                  {k} ({MODELS[k].id})
                </option>
              ))}
            </select>
          </label>{" "}
          <button data-testid="run" onClick={() => runBenchmark(modelKey)}>
            Run benchmark
          </button>
        </>
      )}
      <p data-testid="progress">{progress}</p>
      <pre data-testid="log" style={{ background: "#f4f4f4", padding: 8 }}>
        {logLines.join("\n")}
      </pre>
      <h2>Result</h2>
      <pre data-testid="result" style={{ background: "#eef6ee", padding: 8, whiteSpace: "pre-wrap" }}>
        {resultJson || "(not run yet)"}
      </pre>
    </main>
  );
}
