// 把 trace.json 逐帧画出来，用 WebCodecs（H.264）+ mediabunny 编码成 MP4。
// 只在开发服务器上由 scripts/build-explainer.mjs 打开，结果放到 window.__video

import { BufferTarget, CanvasSource, Mp4OutputFormat, Output } from 'mediabunny';
import { HEIGHT, WIDTH, drawFrame } from './scenes';
import { buildTimeline, transcript, type Trace } from './timeline';

const FPS = 30;
const BITRATE = 300_000; // 画面大多是静止的文字，300 kbps 够清楚；4 MiB ÷ 90 秒 ≈ 370 kbps 是上限

interface RenderResult { mp4: string; duration: number; transcript: string; scenes: { id: number; name: string; mid: number }[] }

declare global {
  interface Window {
    __video?: RenderResult | { error: string }; // record.ts 已经用了 __result
    __still?: (time: number) => string; // --stills 用：画出某一时间点，返回 PNG 的 data URL
  }
}

const base64 = (buf: ArrayBuffer) => new Promise<string>((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve((reader.result as string).slice((reader.result as string).indexOf(',') + 1));
  reader.onerror = () => reject(reader.error);
  reader.readAsDataURL(new Blob([buf]));
});

async function render(): Promise<RenderResult> {
  const trace: Trace = await (await fetch('trace.json')).json();
  const tl = buildTimeline(trace);
  const canvas = Object.assign(document.createElement('canvas'), { width: WIDTH, height: HEIGHT });
  const g = canvas.getContext('2d')!;

  const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
  const source = new CanvasSource(canvas, { codec: 'avc', bitrate: BITRATE });
  output.addVideoTrack(source, { frameRate: FPS });
  await output.start();
  const frames = Math.round(tl.duration * FPS);
  for (let i = 0; i < frames; i++) {
    drawFrame(g, trace, tl, i / FPS);
    await source.add(i / FPS, 1 / FPS);
    if (i % (10 * FPS) === 0) console.log(`第 ${i / FPS} 秒 / ${tl.duration.toFixed(1)} 秒`);
  }
  await output.finalize();

  window.__still = (time) => {
    drawFrame(g, trace, tl, time);
    return canvas.toDataURL('image/png');
  };
  return {
    mp4: await base64(output.target.buffer!),
    duration: frames / FPS,
    transcript: transcript(trace),
    scenes: tl.scenes.map((s) => ({ id: s.id, name: s.name, mid: (s.start + s.end) / 2 })),
  };
}

render().then(
  (result) => (window.__video = result),
  (err) => (window.__video = { error: err instanceof Error ? err.message : String(err) }),
);
