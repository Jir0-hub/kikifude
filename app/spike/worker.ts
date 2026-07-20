/// <reference lib="webworker" />
// G0 spike worker: runs the ASR pipeline off the main thread.
// Disposable spike code — measurement correctness > polish.
import {
  pipeline,
  type AutomaticSpeechRecognitionPipeline,
} from "@huggingface/transformers";

type Dtype = "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "q4f16" | "bnb4" | "auto";
type DtypeMap = Record<string, Dtype>;

interface LoadRequest {
  type: "load";
  modelId: string;
  dtype: DtypeMap;
}

interface TranscribeRequest {
  type: "transcribe";
  audio: Float32Array;
  durationSec: number;
  chunkLengthS: number;
  strideLengthS: number;
  returnTimestamps: boolean;
}

type Request = LoadRequest | TranscribeRequest;

let asr: AutomaticSpeechRecognitionPipeline | null = null;
let loadedModelId: string | null = null;

async function handleLoad(msg: LoadRequest) {
  if (asr && loadedModelId !== msg.modelId) {
    await asr.dispose();
    asr = null;
    loadedModelId = null;
  }
  if (asr) {
    self.postMessage({ type: "loaded", cached: true, metrics: null });
    return;
  }

  const t0 = performance.now();
  // Any progress event implies network/disk fetch still in flight; the gap
  // between the last such event and pipeline() resolving is session init time.
  // KNOWN LIMITATION (observed): file-level events keep firing until just
  // before ready, so initMs measures ~0 and session init is folded into dlMs.
  // loadTotalMs is accurate; warm-cache runs (~4s) approximate init cost.
  let lastFetchEventAt = t0;
  const fileBytes = new Map<string, number>();

  asr = (await pipeline("automatic-speech-recognition", msg.modelId, {
    device: "webgpu",
    dtype: msg.dtype,
    progress_callback: (p: any) => {
      lastFetchEventAt = performance.now();
      if (p.status === "progress" && typeof p.total === "number") {
        fileBytes.set(p.file, p.total);
      }
      if (p.status === "progress") {
        self.postMessage({
          type: "progress",
          file: p.file,
          progress: p.progress,
          loaded: p.loaded,
          total: p.total,
        });
      }
    },
  })) as AutomaticSpeechRecognitionPipeline;
  const tReady = performance.now();
  loadedModelId = msg.modelId;

  // Warmup: 1s of silence. First inference triggers WebGPU shader compilation;
  // measured separately so the real run reflects steady-state speed.
  const tWarm0 = performance.now();
  await asr(new Float32Array(16000), { language: "ja", task: "transcribe" });
  const tWarm1 = performance.now();

  let totalBytes = 0;
  for (const b of fileBytes.values()) totalBytes += b;

  self.postMessage({
    type: "loaded",
    cached: false,
    metrics: {
      dlMs: Math.round(lastFetchEventAt - t0),
      initMs: Math.round(tReady - lastFetchEventAt),
      loadTotalMs: Math.round(tReady - t0),
      warmupMs: Math.round(tWarm1 - tWarm0),
      totalBytes,
    },
  });
}

async function handleTranscribe(msg: TranscribeRequest) {
  if (!asr) throw new Error("Model not loaded");
  const t0 = performance.now();
  const output = await asr(msg.audio, {
    language: "ja",
    task: "transcribe",
    chunk_length_s: msg.chunkLengthS,
    stride_length_s: msg.strideLengthS,
    return_timestamps: msg.returnTimestamps,
  });
  const t1 = performance.now();

  const processMs = Math.round(t1 - t0);
  const rtf = processMs / 1000 / msg.durationSec;
  const single = Array.isArray(output) ? output[0] : output;
  self.postMessage({
    type: "result",
    processMs,
    rtf,
    projected10minSec: Math.round(rtf * 600),
    text: single.text,
    chunks: (single as { chunks?: unknown }).chunks ?? null,
  });
}

self.addEventListener("message", async (e: MessageEvent<Request>) => {
  try {
    if (e.data.type === "load") await handleLoad(e.data);
    else if (e.data.type === "transcribe") await handleTranscribe(e.data);
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? `${err.message}\n${err.stack}` : String(err),
    });
  }
});

self.postMessage({ type: "worker-ready" });
