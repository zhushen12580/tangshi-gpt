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

test('键盘读数与自定义诗', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const select = page.getByTestId('heatmap-select');
  await select.selectOption({ label: '登鹳雀楼' });
  const thumbs = page.getByTestId('heatmap-thumb');
  const main = page.getByTestId('heatmap-main');
  const readout = page.getByTestId('heatmap-readout');
  await thumbs.last().focus();
  await page.keyboard.press('Tab');
  await expect(main).toBeFocused();
  await expect(readout).toHaveText(/^「<s>」看「<s>」：1\.00$/);
  for (let i = 0; i < 10; i++) await page.keyboard.press('ArrowDown');
  for (let i = 0; i < 7; i++) await page.keyboard.press('ArrowRight');
  await expect(readout).toHaveText(/^「海」看「黄」：\d\.\d\d$/);
  // 列不能超过行，行不能超过 23
  for (let i = 0; i < 30; i++) await page.keyboard.press('ArrowRight');
  await expect(readout).toHaveText(/^「海」看「海」：\d\.\d\d$/);
  const box = (await main.boundingBox())!;
  await main.hover({ position: { x: box.width * 1.5 / 24, y: box.height * 2.5 / 24 } });
  await expect(readout).toHaveText(/^「日」看「白」：\d\.\d\d$/);

  await page.keyboard.press('Shift+Tab');
  await expect(thumbs.last()).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(thumbs.last()).toHaveAttribute('aria-pressed', 'true');
  await expect(thumbs.first()).toHaveAttribute('aria-pressed', 'false');

  const input = page.getByTestId('heatmap-input');
  const view = page.getByRole('button', { name: '查看' });
  const x = page.getByTestId('heatmap-axis-x').locator('span');
  const error = page.getByTestId('heatmap-error');
  await input.fill('床前明月光，疑是地上霜。举头望明月，低头思故乡。');
  await view.click();
  await expect(x.nth(1)).toHaveText('床');
  await expect(x.nth(3)).toHaveText('明');
  await expect(select.locator('option:checked')).toHaveText('自定义');
  await expect(error).toHaveText('');

  await input.fill('床前明月光疑是地上霜举头望明月低头思ok');
  await view.click();
  await expect(error).toHaveText('以下字符不在字表中：o、k');
  await expect(x.nth(1)).toHaveText('床');
  await input.fill('床前明月光');
  await input.press('Enter');
  await expect(error).toHaveText('需要恰好 20 个汉字（当前 5 个）');
  await expect(x.nth(1)).toHaveText('床');
  await expect(select.locator('option:checked')).toHaveText('自定义');

  await select.selectOption({ label: '春晓' });
  await expect(x.nth(1)).toHaveText('春');
  await select.selectOption({ label: '自定义' });
  await expect(x.nth(1)).toHaveText('床');
});
