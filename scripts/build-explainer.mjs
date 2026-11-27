// 原理视频的生成脚本，需要有 GPU 加速的机器，不在 CI 里跑。
// 用法：npm run build:explainer-trace [-- --backend=<名字>]
//   在 Vite 开发服务器（5174）上打开 explainer/record.html，录一次 1500 步的真实训练，写 explainer/trace.json。
//   --backend 原样拼到录制页的 ?backend=，用来人工验证 CPU 下的失败路径。
// 页面在 Playwright 的 Chromium 里跑；channel: 'chromium' 是新 headless，能用上真实 GPU。
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { createServer } from 'vite';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const PORT = 5174;
const RECORD_TIMEOUT = 60 * 60_000; // WebGPU 约 7 步/秒，1500 步加 31 次里程碑约 5 分钟；软件 WebGL 要半小时以上
const LABEL = { webgpu: 'WebGPU', webgl: 'WebGL' };

/** 启动开发服务器和浏览器，打开 explainer 下的页面交给 fn；结束时（包括出错时）都关掉 */
async function withPage(path, fn) {
  const server = await createServer({ root: ROOT, server: { port: PORT, strictPort: true } });
  let browser;
  try {
    await server.listen();
    browser = await chromium.launch({ channel: 'chromium' });
    const page = await browser.newPage();
    page.on('console', (msg) => console.log(msg.text()));
    page.on('pageerror', (err) => console.error(err));
    await page.goto(`http://localhost:${PORT}/explainer/${path}`);
    return await fn(page);
  } finally {
    await browser?.close();
    await server.close();
  }
}

async function recordTrace(args) {
  const backend = args.find((a) => a.startsWith('--backend='))?.slice('--backend='.length);
  const result = await withPage(`record.html${backend === undefined ? '' : `?backend=${backend}`}`, async (page) => {
    await page.waitForFunction(() => window.__result, null, { timeout: RECORD_TIMEOUT, polling: 1000 });
    return page.evaluate(() => window.__result);
  });
  // 页面只在 traceErrors 为空时才给 trace；出错或验证 loss 没有回升时不写文件，原来的 trace.json 保持不变
  if (!result.trace) throw new Error(result.error);
  const { meta, best } = result.trace;
  await writeFile(join(ROOT, 'explainer', 'trace.json'), JSON.stringify(result.trace) + '\n');
  console.log(`录制完成：${meta.steps} 步，后端 ${LABEL[meta.backend]}，验证 loss 最低在第 ${best.step} 步`);
}

const [command, ...args] = process.argv.slice(2);
try {
  if (command !== 'trace') throw new Error('用法：node scripts/build-explainer.mjs trace [--backend=<名字>]');
  await recordTrace(args);
} catch (err) {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}
