import { expect, test } from '@playwright/test';
import { openApp, waitReady } from './helpers';

test('没有 WebGPU 时降级到 WebGL', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { get: () => undefined }));
  await openApp(page);
  await waitReady(page);
  await expect(page.getByTestId('backend')).toHaveText('WebGL');
  await expect(page.getByTestId('backend-notice')).toContainText('WebGPU 不可用，已改用 WebGL');
});

test('数据集加载失败时可以重试', async ({ page }) => {
  await page.route('**/data/poems.json', (r) => r.fulfill({ status: 404, body: '' }));
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
