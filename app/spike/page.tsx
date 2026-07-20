"use client";
// G0 spike page: benchmarks kotoba-whisper ONNX (and fallback model) on WebGPU.
// Results are rendered on screen and logged as one-line JSON ("SPIKE_RESULT ...")
// so an automated harness can scrape them from the console.
import { useEffect, useRef, useState } from "react";
import { waitForMessage, type WorkerMsg } from "@/lib/worker-client";

// approxMB = sum of the ONNX files actually fetched for the given dtype map
// (encoder_model_<dtype>.onnx + decoder_model_merged_<dtype>.onnx), read from
// the HF file tree. Sanity check: turbo computes to 1468MB and the 2026-07-20
// desktop run measured 1.47GB, so the figures line up with reality.
//
// Why the light models use fp16/fp16 instead of the turbo recipe (fp16 + q4f16):
// onnx-community/whisper-{tiny,base,small} DO NOT ship q4f16 files at all — only
// fp32 / fp16 / q4 / int8 / uint8 / bnb4. Copying the turbo dtype map here would
// 404 at load time and the device test would be a false negative. fp16 is also
// the smallest option for tiny/base: their decoders are dominated by the token
// embedding table, which q4 leaves in fp32, so decoder q4 is LARGER than fp16
// (tiny: q4 86.7MB vs fp16 59.6MB). fp16 keeps the size ladder monotonic and
// avoids stacking quantization damage on models whose Japanese accuracy is
// already marginal — this run is measuring the memory ceiling, not quality.
const MODELS: Record<
  string,
  { id: string; dtype: Record<string, string>; approxMB?: number }
> = {
  // --- Memory ladder for the iPhone XS OOM investigation (2026-07-20) ---
  // iPhone XS (A12 / RAM 4GB) acquires a WebGPU adapter but the tab is killed by
  // jetsam partway through loading turbo (~1.47GB), with no catchable exception.
  // The binding constraint is memory, not WebGPU support. These rungs exist to
  // bisect the ceiling in a single device session: run tiny -> base -> small and
  // note the first key that crashes.
  tiny: {
    id: "onnx-community/whisper-tiny",
    dtype: { encoder_model: "fp16", decoder_model_merged: "fp16" },
    approxMB: 76, // encoder 16.5MB + decoder_merged 59.6MB
  },
  // --- fp16 hypothesis probe, PRIMARY series (Safari / WebKit, 2026-07-20) ---
  // Mac mini M2 / Safari 26.3.1 ran `tiny` (39M params, fp16) at RTF 1.018 with
  // warmupMs 13,310, while the same machine on Chrome 145 ran `turbo` (809M
  // params, 20x larger) at RTF 0.137 / warmupMs 2,829. A 20x smaller model being
  // 7x slower is not a capacity problem.
  //
  // The original suspicion — that WebKit lacks `shader-f16` and has to convert
  // every fp16 tensor — is REFUTED: the Safari machine reports `shader-f16: YES`
  // and a maxBufferSize within 4 bytes of Chrome's. Do not resurrect it.
  //
  // These rungs are kept anyway. They isolate dtype itself (same model, same
  // audio, formats containing no f16 at all), which now answers a different
  // question: whether dtype handling matters here for some reason OTHER than
  // the missing-feature story. If `tiny-fp32` is dramatically faster than `tiny`
  // despite being 2x the download, dtype is still implicated and the cause lies
  // in how f16 is executed rather than whether it is supported.
  //
  // These must be `tiny`, not `small`. On the Mac mini M2 / Safari 26.3.1 under
  // investigation, `small` (485MB) loads but is killed during inference, so it
  // cannot produce a baseline — and `small-fp32` is twice its size, so it cannot
  // produce a verdict either. `tiny` is the only rung with a completed run on
  // that machine (RTF 1.018 / warmupMs 13,310), which makes it the only place a
  // dtype A/B can actually be compared. Sizes here stay far under the ceiling.
  //
  // Verified against the HF file tree before listing (a 404 would surface as a
  // load failure and be misread as an OOM crash): q8 -> encoder_model_quantized
  // .onnx + decoder_model_merged_quantized.onnx, fp32 -> the unsuffixed
  // encoder_model.onnx + decoder_model_merged.onnx. Neither contains any f16.
  "tiny-q8": {
    id: "onnx-community/whisper-tiny",
    dtype: { encoder_model: "q8", decoder_model_merged: "q8" },
    approxMB: 41, // encoder 10.1MB + decoder_merged 30.7MB
  },
  "tiny-fp32": {
    id: "onnx-community/whisper-tiny",
    dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" },
    approxMB: 152, // encoder 32.9MB + decoder_merged 118.6MB
  },
  base: {
    id: "onnx-community/whisper-base",
    dtype: { encoder_model: "fp16", decoder_model_merged: "fp16" },
    approxMB: 146, // encoder 41.3MB + decoder_merged 104.7MB
  },
  small: {
    id: "onnx-community/whisper-small",
    dtype: { encoder_model: "fp16", decoder_model_merged: "fp16" },
    approxMB: 485, // encoder 176.6MB + decoder_merged 308.6MB
  },
  // --- Same dtype A/B at `small` scale, for the CHROME side of the comparison ---
  // Kept deliberately, but NOT for the Safari machine: `small` already OOMs there
  // mid-inference. Their purpose is to establish what the dtype spread looks like
  // on a browser that runs this workload well, so the Safari `tiny-*` numbers can
  // be read against a known-good control instead of in isolation. Run these on
  // desktop Chrome. (q4f16 does NOT exist for tiny/base/small — see note above.)
  "small-q8": {
    id: "onnx-community/whisper-small",
    dtype: { encoder_model: "q8", decoder_model_merged: "q8" },
    approxMB: 249, // encoder 92.3MB + decoder_merged 156.8MB
  },
  "small-fp32": {
    id: "onnx-community/whisper-small",
    dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" },
    approxMB: 968, // encoder 352.8MB + decoder_merged 615.3MB
  },
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
    approxMB: 1468, // encoder 1274.3MB + decoder_merged 193.5MB
  },
};

const fmtModel = (k: string) => {
  const m = MODELS[k];
  return `${k} — ${m.id}${m.approxMB ? ` (~${m.approxMB}MB)` : ""}`;
};

// Limits worth recording for the memory/fp16 investigation. GPUSupportedLimits
// exposes its values as prototype getters, so Object.entries() on it returns {}
// — the names have to be listed explicitly.
const LIMIT_KEYS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
] as const;

interface GpuCaps {
  shaderF16: boolean;
  features: string[];
  limits: Record<string, number | null>;
}

// Reads the adapter capabilities that the Safari-vs-Chrome comparison hinges on.
// `shader-f16` is surfaced first only because it was the original suspect; it was
// measured as present on BOTH browsers (2026-07-20), so it explains nothing. The
// full `features` list is the part that still earns its keep: diffing the two
// lists is what narrowed the remaining candidates. See SPIKE.md for the current
// (unverified) hypothesis and what has already been ruled out.
function readCaps(adapter: GPUAdapter): GpuCaps {
  const limits: Record<string, number | null> = {};
  for (const k of LIMIT_KEYS) {
    const v: unknown = adapter.limits[k];
    limits[k] = typeof v === "number" ? v : null;
  }
  return {
    shaderF16: adapter.features.has("shader-f16"),
    features: [...adapter.features].sort(),
    limits,
  };
}

export default function SpikePage() {
  const workerRef = useRef<Worker | null>(null);
  const runningRef = useRef(false);
  const [gpuInfo, setGpuInfo] = useState<string>("checking...");
  const [gpuCaps, setGpuCaps] = useState<GpuCaps | null>(null);
  const [webgpuOk, setWebgpuOk] = useState<boolean | null>(null);
  const [modelKey, setModelKey] = useState("kotoba");
  const [logLines, setLogLines] = useState<string[]>([]);
  const [progress, setProgress] = useState<string>("");
  // Shown prominently while a run is in flight. On iOS the tab can be killed by
  // jetsam with no exception and no console, so the only surviving evidence of
  // "which size was being loaded" is what was on screen — screenshot it.
  const [activeKey, setActiveKey] = useState<string>("");
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
        setGpuCaps(readCaps(adapter));
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
    setActiveKey(key);
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

      log(
        `load start: ${cfg.id} dtype=${JSON.stringify(cfg.dtype)} approx=${cfg.approxMB ?? "?"}MB`,
      );
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
        approxMB: cfg.approxMB ?? null,
        chunkLengthS,
        strideLengthS,
        returnTimestamps,
        maxSec,
        src,
        gpu: gpuInfo,
        // Adapter capabilities travel with the measurement on purpose: RTF alone
        // cannot distinguish "slow GPU" from "missing GPU capability", and the two
        // runs being compared happen on different machines/browsers.
        gpuCaps,
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
      {gpuCaps ? (
        <div
          data-testid="gpu-caps"
          style={{
            background: gpuCaps.shaderF16 ? "#eef6ee" : "#fff4d6",
            padding: 8,
            marginBottom: 8,
          }}
        >
          <p style={{ margin: 0, fontWeight: "bold" }}>
            shader-f16: {gpuCaps.shaderF16 ? "YES" : "NO"}
          </p>
          <p style={{ margin: "4px 0 0" }}>features: {gpuCaps.features.join(", ") || "(none)"}</p>
          <p style={{ margin: "4px 0 0" }}>
            {LIMIT_KEYS.map((k) => `${k}=${gpuCaps.limits[k] ?? "?"}`).join(" / ")}
          </p>
        </div>
      ) : null}
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
                  {fmtModel(k)}
                </option>
              ))}
            </select>
          </label>{" "}
          <button data-testid="run" onClick={() => runBenchmark(modelKey)}>
            Run benchmark
          </button>
        </>
      )}
      {activeKey ? (
        <p
          data-testid="active-model"
          style={{ fontWeight: "bold", background: "#fff4d6", padding: 8 }}
        >
          running: {fmtModel(activeKey)}
        </p>
      ) : null}
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
