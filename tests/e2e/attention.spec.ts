import { expect, test } from '@playwright/test';
import { openApp, waitForStep, waitReady } from './helpers';

test('预设诗与小图', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const select = page.getByTestId('heatmap-select');
  await expect(select.locator('option:checked')).toHaveText('登鹳雀楼');
  await expect(page.getByTestId('heatmap-step')).toHaveText('来自第 0 步的模型');
  const thumbs = page.getByTestId('heatmap-thumb');
  await expect(thumbs).toHaveText(Array.from({ length: 12 }, (_, i) => `第${Math.floor(i / 4) + 1}层·头${i % 4 + 1}`));
  const labels = ['<s>', ...'白日依山尽，黄河入海流。欲穷千里目，更上一层楼'];
  const x = page.getByTestId('heatmap-axis-x').locator('span');
  await expect(x).toHaveText(labels);
  await expect(page.getByTestId('heatmap-axis-y').locator('span')).toHaveText(labels);

  const pixels = await page.getByTestId('heatmap-main').evaluate((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d')!;
    const center = (col: number) => Math.floor((col + 0.5) * canvas.width / 24);
    return [0, 23].map((col) => Array.from(ctx.getImageData(center(col), center(0), 1, 1).data));
  });
  [[31, 26, 23, 255], [245, 239, 224, 255]].forEach((color, i) => {
    color.forEach((channel, j) => expect(Math.abs(pixels[i][j] - channel)).toBeLessThanOrEqual(3));
  });

  await thumbs.nth(5).click();
  await expect(thumbs.nth(5)).toHaveAttribute('aria-pressed', 'true');
  await expect(thumbs.first()).toHaveAttribute('aria-pressed', 'false');
  await select.selectOption({ label: '春晓' });
  await expect(x.nth(1)).toHaveText('春');

  await page.getByTestId('start-toggle').click();
  await waitForStep(page, 50);
  await expect.poll(() => page.getByTestId('heatmap-step').textContent(), { timeout: 60_000 })
    .toBe('来自第 50 步的模型');
  await expect(x.nth(1)).toHaveText('春');
  await page.getByTestId('start-toggle').click();
  await expect(page.getByTestId('start-toggle')).toHaveText('继续', { timeout: 60_000 });
});
