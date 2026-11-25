import { expect, test } from '@playwright/test';
import { num, openApp, waitReady } from './helpers';

test('加载状态与页头', async ({ page }) => {
  // 让数据晚到，才能看到加载中的状态
  await page.route('**/data/poems.json', (r) => setTimeout(() => r.continue(), 1500));
  await openApp(page, '?backend=webgl');

  const btn = page.getByTestId('start-toggle');
  await expect(btn).toBeDisabled();
  await expect(btn).toHaveText('正在准备…');

  await waitReady(page);

  const info = (await page.getByTestId('dataset-info').textContent()) ?? '';
  const m = info.match(/^数据集：(\d+) 首五言绝句 · 字表 (\d+) 字 · 来自 chinese-poetry（MIT）$/);
  expect(m, info).not.toBeNull();
  const [count, vocabSize] = [Number(m![1]), Number(m![2])];
  expect(count).toBeGreaterThanOrEqual(3000);
  expect(Math.abs((await num(page, 'loss-baseline')) - Math.log(vocabSize))).toBeLessThan(0.001);
});
