import { expect, test, type Page } from '@playwright/test';
import { openApp, waitForStep, waitReady } from './helpers';

test.use({ viewport: { width: 1280, height: 800 } });

// 通俗版《静夜思》：不在预设里，和数据集收录的版本也不同
const JINGYESI = '床前明月光，疑是地上霜。举头望明月，低头思故乡。';
const JINGYESI_LABELS = ['<s>', ...'床前明月光，疑是地上霜。举头望明月，低头思故乡'];

const shot = (page: Page, name: string) =>
  page.screenshot({ path: `test-results/shots/${name}.png`, fullPage: true });

test('真实流程', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await openApp(page, '?backend=webgl');
  await waitReady(page);

  // 只用键盘：Tab 到「开始训练」，Enter 开始，再按 Enter 暂停
  const btn = page.getByTestId('start-toggle');
  for (let i = 0; i < 5 && !(await btn.evaluate((el) => el === document.activeElement)); i++) {
    await page.keyboard.press('Tab');
  }
  await expect(btn).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(btn).toHaveText('暂停');
  await waitForStep(page, 50);
  await page.keyboard.press('Enter');
  await expect(btn).toHaveText('继续', { timeout: 60_000 });

  const items = page.getByTestId('sample-item');
  expect(await items.count()).toBeGreaterThanOrEqual(2);
  for (const item of await items.all()) {
    const lines = await item.locator('.poem-line').allTextContents();
    expect(lines).toHaveLength(4);
    for (const line of lines) expect(line).toMatch(/^[一-鿿]{5}[，。]$/);
  }

  await page.getByTestId('heatmap-input').fill(JINGYESI);
  await page.getByRole('button', { name: '查看' }).click();
  await expect(page.getByTestId('heatmap-axis-x').locator('span')).toHaveText(JINGYESI_LABELS);
  await expect(page.getByTestId('heatmap-axis-y').locator('span')).toHaveText(JINGYESI_LABELS);
  const readout = page.getByTestId('heatmap-readout');
  await page.getByTestId('heatmap-main').focus();
  await expect(readout).toHaveText(/^「<s>」看「<s>」：1\.00$/);
  await page.keyboard.press('ArrowDown');
  await expect(readout).toHaveText(/^「床」看「<s>」：\d\.\d\d$/);
  await page.keyboard.press('ArrowRight');
  await expect(readout).toHaveText(/^「床」看「床」：\d\.\d\d$/);

  const slider = page.getByTestId('temperature');
  await slider.focus();
  for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowLeft');
  await expect(page.getByTestId('temperature-value')).toHaveText('0.3');
  const count = await items.count();
  await page.getByTestId('generate').focus();
  await page.keyboard.press('Enter');
  await expect(items).toHaveCount(count + 1, { timeout: 60_000 });
  await expect(items.first().locator('.sample-meta')).toContainText('手动 · 温度 0.3 · 约束');
  expect(errors).toEqual([]);
});

test('空输入、错误与重试成功', async ({ page }) => {
  await page.route('**/data/poems.json', (r) => r.fulfill({ status: 404, body: '' }));
  await openApp(page, '?backend=webgl');
  await expect(page.getByTestId('data-error')).toContainText('数据集加载失败');
  await expect(page.getByTestId('start-toggle')).toBeDisabled();
  await page.unroute('**/data/poems.json');
  await page.getByTestId('retry').click();
  await waitReady(page);

  const input = page.getByTestId('heatmap-input');
  const view = page.getByRole('button', { name: '查看' });
  const error = page.getByTestId('heatmap-error');
  await view.click();
  await expect(error).toHaveText('需要恰好 20 个汉字（当前 0 个）');
  await input.fill(JINGYESI);
  await view.click();
  await expect(error).toHaveText('');
  await expect(page.getByTestId('heatmap-axis-x').locator('span')).toHaveText(JINGYESI_LABELS);

  await page.getByTestId('start-toggle').click();
  await waitForStep(page, 1, 60_000);
  await page.getByTestId('start-toggle').click();
  await expect(page.getByTestId('start-toggle')).toHaveText('继续', { timeout: 60_000 });
});

test('热力图轴标签不重叠，焦点框不压字', async ({ page }) => {
  for (const width of [375, 1280]) {
    await page.setViewportSize({ width, height: 800 });
    await openApp(page, '?backend=webgl');
    await waitReady(page);
    await page.getByTestId('heatmap-thumb').last().focus();
    await page.keyboard.press('Tab'); // 键盘聚焦，:focus-visible 才生效
    await expect(page.getByTestId('heatmap-main')).toBeFocused();
    // 正数 = 重叠的像素
    const overlaps = await page.evaluate(() => {
      const textRect = (el: Element) => {
        const range = document.createRange();
        range.selectNodeContents(el);
        return range.getBoundingClientRect();
      };
      const spans = (id: string) => [...document.querySelectorAll(`[data-testid="${id}"] span`)].map(textRect);
      const xs = spans('heatmap-axis-x');
      const ys = spans('heatmap-axis-y');
      const main = document.querySelector('[data-testid="heatmap-main"]')!;
      const m = main.getBoundingClientRect();
      const style = getComputedStyle(main);
      const ring = parseFloat(style.outlineOffset) + parseFloat(style.outlineWidth);
      return {
        neighbours: Math.max(...xs.slice(1).map((r, i) => xs[i].right - r.left)),
        ringTop: Math.max(...xs.map((r) => r.bottom)) - (m.top - ring),
        ringLeft: Math.max(...ys.map((r) => r.right)) - (m.left - ring),
      };
    });
    for (const [k, v] of Object.entries(overlaps)) expect.soft(v, `${width}px ${k}`).toBeLessThanOrEqual(0);
  }
});

test('各状态截图', async ({ page }) => {
  const scrollWidth = () => page.evaluate(() => document.documentElement.scrollWidth);
  await page.route('**/data/poems.json', async (r) => {
    await new Promise((resolve) => setTimeout(resolve, 3000));
    await r.continue();
  });
  await openApp(page, '?backend=webgl');
  await expect(page.getByTestId('start-toggle')).toHaveText('正在准备…');
  await shot(page, 'loading');
  await waitReady(page);
  await page.keyboard.press('Tab'); // 让截图里能看到焦点框
  await shot(page, 'ready');

  const btn = page.getByTestId('start-toggle');
  await btn.click();
  await waitForStep(page, 60);
  await shot(page, 'training');
  await page.setViewportSize({ width: 375, height: 667 });
  expect(await scrollWidth()).toBeLessThanOrEqual(375);
  await shot(page, 'narrow-training');
  await page.setViewportSize({ width: 1280, height: 800 });
  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 60_000 });
  await shot(page, 'paused');

  const main = page.getByTestId('heatmap-main');
  await main.focus();
  for (let i = 0; i < 8; i++) await page.keyboard.press('ArrowDown');
  for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowRight');
  await shot(page, 'heatmap-focus');

  await page.unroute('**/data/poems.json');
  await page.route('**/data/poems.json', (r) => r.fulfill({ status: 404, body: '' }));
  await page.reload();
  await expect(page.getByTestId('data-error')).toBeVisible();
  await shot(page, 'data-error');

  await page.unroute('**/data/poems.json');
  await page.setViewportSize({ width: 375, height: 667 });
  await page.reload();
  await waitReady(page);
  expect(await scrollWidth()).toBeLessThanOrEqual(375);
  await shot(page, 'narrow-ready');
});
