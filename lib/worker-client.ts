// Shared client-side helpers for talking to the ASR worker.

export interface WorkerMsg {
  type: string;
  [key: string]: unknown;
}

/**
 * Resolves with the first message whose type is in `types`;
 * rejects if an "error" message arrives first.
 */
export function waitForMessage(
  worker: Worker,
  types: string[],
): Promise<WorkerMsg> {
  return new Promise((resolve, reject) => {
    const handler = (e: MessageEvent<WorkerMsg>) => {
      if (e.data.type === "error") {
        worker.removeEventListener("message", handler);
        reject(new Error(String(e.data.message)));
      } else if (types.includes(e.data.type)) {
        worker.removeEventListener("message", handler);
        resolve(e.data);
      }
    };
    worker.addEventListener("message", handler);
  });
}

/**
 * Decodes an audio file/blob to 16 kHz mono Float32Array
 * (decodeAudioData resamples to the context sample rate).
 */
export async function decodeTo16kMono(
  data: ArrayBuffer,
): Promise<{ audio: Float32Array; durationSec: number }> {
  const ctx = new OfflineAudioContext(1, 16000, 16000);
  const decoded = await ctx.decodeAudioData(data);
  return { audio: decoded.getChannelData(0), durationSec: decoded.duration };
}
