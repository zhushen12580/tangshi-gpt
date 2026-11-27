import { expect, test } from '@playwright/test';
import { num, openApp, waitReady } from './helpers';

test('没有 WebGPU 时降级到 WebGL', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { get: () => undefined }));
  await openApp(page);
  await waitReady(page);
  await expect(page.getByTestId('backend')).toHaveText('WebGL');
  await expect(page.getByTestId('backend-notice')).toContainText('WebGPU 不可用，已改用 WebGL');
  await expect(page.getByTestId('backend-notice')).not.toContainText('未知后端'); // 没写 ?backend= 不算未知值
});

// 404 和「HTTP 200 但数据不能用」都要走重试
const BAD_RESPONSES = [
  { name: '404', fulfill: { status: 404, body: '' } },
  { name: '空数据集', fulfill: { json: { poems: [] } } },
];

for (const { name, fulfill } of BAD_RESPONSES) test(`数据集加载失败时可以重试（${name}）`, async ({ page }) => {
  await page.route('**/data/poems.json', (r) => r.fulfill(fulfill));
  await openApp(page, '?backend=webgl');
  await expect(page.getByTestId('data-error')).toContainText('数据集加载失败');
  const retry = page.getByTestId('retry');
  await expect(retry).toHaveText('重试');
  await expect(page.getByTestId('start-toggle')).toBeDisabled();

  await page.unroute('**/data/poems.json');
  await retry.click();
  await waitReady(page);
  await expect(page.getByTestId('data-error')).toBeHidden();
});

test('未知后端按自动模式选择并提示', async ({ page }) => {
  await openApp(page, '?backend=foo');
  await waitReady(page);
  await expect(page.getByTestId('backend-notice')).toContainText('未知后端 foo，已自动选择');
  await expect(page.getByTestId('backend')).toHaveText(/^(WebGPU|WebGL|CPU)$/);
});

test('写了 ?backend= 但值为空：按自动模式选择并提示', async ({ page }) => {
  await openApp(page, '?backend=');
  await waitReady(page);
  await expect(page.getByTestId('backend-notice')).toContainText('未知后端（空值），已自动选择');
  await expect(page.getByTestId('backend')).toHaveText(/^(WebGPU|WebGL|CPU)$/);
});

test('很长的未知后端值在窄屏下折行，不出现横向滚动', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await openApp(page, '?backend=' + 'x'.repeat(500));
  await waitReady(page);
  await expect(page.getByTestId('backend-notice')).toContainText('，已自动选择');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
});

test('只用 CPU 时提示非常慢，仍能训练', async ({ page }) => {
  await openApp(page, '?backend=cpu');
  await waitReady(page);
  await expect(page.getByTestId('backend')).toHaveText('CPU');
  await expect(page.getByTestId('backend-notice')).toContainText('未使用 GPU 加速，训练会非常慢');
  // CPU 下训练占满主线程，Playwright 的每条命令都要等主线程空出来才执行，会多算好几步，所以在页面里点击并计时
  const ms = await page.evaluate(() => new Promise<number>((resolve) => {
    const step = document.querySelector('[data-testid="step"]')!;
    const t0 = performance.now();
    new MutationObserver(() => Number(step.textContent) >= 1 && resolve(performance.now() - t0))
      .observe(step, { childList: true, characterData: true, subtree: true });
    document.querySelector<HTMLButtonElement>('[data-testid="start-toggle"]')!.click();
  }));
  console.log(`CPU：点击到步数 ≥ 1 用时 ${(ms / 1000).toFixed(1)} 秒`);
  expect(ms).toBeLessThan(60_000);
});

test('只用 CPU 时，验证中和一步算到一半点「暂停」，都在 2 秒内变成「继续」，步数停住', async ({ page }) => {
  const errors: Error[] = [];
  page.on('pageerror', (e) => errors.push(e));
  await openApp(page, '?backend=cpu');
  await waitReady(page);
  const box = (await page.getByTestId('start-toggle').boundingBox())!;
  // 排在应用自己的监听器之后：记下点击处理完的时刻、按钮文字和当时的步数；
  // 另外记下最长的主线程任务（longtask）：点击要等的最多就是正在执行的这一个任务
  await page.evaluate(() => {
    const w = window as unknown as { clicks: object[]; maxTask: number };
    w.clicks = [];
    w.maxTask = 0;
    const btn = document.querySelector('[data-testid="start-toggle"]')!;
    const step = document.querySelector('[data-testid="step"]')!;
    btn.addEventListener('click', () => w.clicks.push({ at: Date.now(), text: btn.textContent, step: step.textContent }));
    new PerformanceObserver((list) => list.getEntries().forEach((e) => (w.maxTask = Math.max(w.maxTask, e.duration))))
      .observe({ type: 'longtask' });
  });
  // 像真人一样用鼠标点：输入事件要等主线程空出来才会被处理，所以从发出点击算到处理完
  let n = 0;
  const click = async () => {
    const t0 = Date.now();
    await page.mouse.click(box.x + 10, box.y + box.height / 2);
    await page.waitForFunction((k) => (window as unknown as { clicks: object[] }).clicks.length >= k, ++n);
    const rec = await page.evaluate(() => (window as unknown as { clicks: { at: number; text: string; step: string }[] }).clicks.at(-1)!);
    return { ...rec, latency: rec.at - t0 };
  };
  const expectQuickPause = async () => {
    const r = await click();
    console.log(`CPU：暂停点击 ${r.latency} ms 后处理完，步数 ${r.step}`);
    expect(r.text).toBe('继续');
    expect(r.latency).toBeLessThan(2000);
    return r;
  };

  expect((await click()).text).toBe('暂停');
  await page.waitForTimeout(750); // 第 0 步的里程碑：CPU 下验证集前向要算半分钟
  expect((await expectQuickPause()).step).toBe('0');

  await expect(page.getByTestId('sample-item')).toHaveCount(1, { timeout: 120_000 }); // 里程碑在后台做完
  // 先完整训练一步，量出 CPU 上一步要多久
  const resumed = await click();
  expect(resumed.text).toBe('暂停');
  await expect.poll(() => num(page, 'step'), { timeout: 60_000 }).toBeGreaterThanOrEqual(1);
  const stepMs = Date.now() - resumed.at;
  console.log(`CPU：一步用时约 ${(stepMs / 1000).toFixed(1)} 秒`);

  await page.waitForTimeout(750); // 下一步正在算
  const paused = await expectQuickPause();
  await page.waitForTimeout(Math.max(0, paused.at + 2000 - Date.now()));
  const settled = await page.getByTestId('step').textContent();
  expect(settled).toBe(paused.step);
  // 等得比一整步还久：如果没算完的那一步没被放弃，这期间步数就会加 1
  await page.waitForTimeout(stepMs + 3000);
  expect(await page.getByTestId('step').textContent()).toBe(settled);

  // 放弃的那一步不影响之后的训练
  expect((await click()).text).toBe('暂停');
  await expect.poll(() => num(page, 'step'), { timeout: 60_000 }).toBeGreaterThanOrEqual(Number(paused.step) + 1);
  await expectQuickPause();
  await page.waitForTimeout(500); // longtask 记录是异步送达的
  const maxTask = await page.evaluate(() => (window as unknown as { maxTask: number }).maxTask);
  console.log(`CPU：最长的主线程任务 ${Math.round(maxTask)} ms`);
  expect(maxTask).toBeGreaterThan(0); // 确实观察到了长任务（CPU 下一行就要几百毫秒），断言不是空转
  expect(maxTask).toBeLessThan(2000);
  expect(errors).toEqual([]);
});
