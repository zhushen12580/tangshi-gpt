import { expect, test } from '@playwright/test';
import { num, openApp, waitForStep, waitReady } from './helpers';

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

test('训练 200 步 loss 下降', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  await expect(page.getByTestId('backend')).toHaveText('WebGL');
  const params = await num(page, 'params');
  expect(params).toBeGreaterThanOrEqual(0.8);
  expect(params).toBeLessThanOrEqual(1.3);

  const empty = page.getByTestId('loss-empty');
  await expect(empty).toBeVisible();
  await expect(empty).toHaveText('点击「开始训练」后这里会画出 loss 曲线');
  const chart = page.getByTestId('loss-chart');
  const pixels = () => chart.evaluate((c: HTMLCanvasElement) => c.toDataURL());
  const blank = await pixels();

  const btn = page.getByTestId('start-toggle');
  await btn.click();
  await expect(btn).toHaveText('暂停');
  const t0 = Date.now();
  await waitForStep(page, 200);
  const secs = (Date.now() - t0) / 1000;
  const speed = await num(page, 'speed');
  console.log(`200 步用时 ${secs.toFixed(1)} 秒，页面显示 ${speed} 步/秒`);
  expect(speed).toBeGreaterThan(0);

  const elapsed = page.getByTestId('elapsed');
  const e1 = (await elapsed.textContent()) ?? '';
  expect(e1).toMatch(/^\d\d:\d\d$/);
  await page.waitForTimeout(2000);
  expect(await elapsed.textContent()).not.toBe(e1);

  const initial = await num(page, 'loss-initial');
  expect(Math.abs(initial - (await num(page, 'loss-baseline')))).toBeLessThanOrEqual(0.5);

  // 第 200 步的里程碑（验证 + 写一首诗）可能还在进行，暂停要等它做完
  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 60_000 });
  expect(await num(page, 'loss-current')).toBeLessThanOrEqual(initial * 0.85);

  await expect(empty).toBeHidden();
  expect(await pixels()).not.toBe(blank);
  // 停在 ≥ 200 步时，第 200 步的验证 loss 已经算出
  expect(await num(page, 'loss-val')).toBeLessThan(initial);
});
