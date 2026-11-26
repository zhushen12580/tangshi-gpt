import { expect, test } from '@playwright/test';
import { openApp, waitReady } from './helpers';

test('没有 WebGPU 时降级到 WebGL', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { get: () => undefined }));
  await openApp(page);
  await waitReady(page);
  await expect(page.getByTestId('backend')).toHaveText('WebGL');
  await expect(page.getByTestId('backend-notice')).toContainText('WebGPU 不可用，已改用 WebGL');
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
