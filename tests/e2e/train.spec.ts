import { expect, test, type Page } from '@playwright/test';
import { holdDataset, num, openApp, waitForStep, waitHeatmap, waitReady } from './helpers';

test('加载状态与页头', async ({ page }) => {
  // 扣住数据，才能看到加载中的状态
  const release = await holdDataset(page);
  await openApp(page, '?backend=webgl');

  const btn = page.getByTestId('start-toggle');
  await expect(btn).toBeDisabled();
  await expect(btn).toHaveText('正在准备…');

  release();
  await waitReady(page);

  const info = (await page.getByTestId('dataset-info').textContent()) ?? '';
  const m = info.match(/^数据集：(\d+) 首五言绝句 · 字表 (\d+) 字 · 来自 chinese-poetry（MIT）$/);
  expect(m, info).not.toBeNull();
  const [count, vocabSize] = [Number(m![1]), Number(m![2])];
  expect(count).toBeGreaterThanOrEqual(3000);
  expect(Math.abs((await num(page, 'loss-baseline')) - Math.log(vocabSize))).toBeLessThan(0.001);
});

test('「开始训练」不等第一张热力图', async ({ page }) => {
  // 按钮变成可用的那一刻记下热力图标题。第一张热力图在软件 WebGL 下要编译一批着色器，
  // 机器忙的时候要十几秒，挡在启动路径上会让「10 秒内可用」超时（release 发现 F-34bc53267ed5）
  await page.addInitScript(() => {
    new MutationObserver((_, mo) => {
      const btn = document.querySelector<HTMLButtonElement>('[data-testid="start-toggle"]');
      if (!btn || btn.disabled) return;
      mo.disconnect();
      (window as unknown as { heatmapAtReady: string | null }).heatmapAtReady =
        document.querySelector('[data-testid="heatmap-step"]')?.textContent ?? null;
    }).observe(document, { subtree: true, attributes: true, attributeFilter: ['disabled'] });
  });
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  expect(await page.evaluate(() => (window as unknown as { heatmapAtReady: string | null }).heatmapAtReady)).toBe('');
  await waitHeatmap(page);
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

  // 第 200 步的里程碑（验证 + 写一首诗）可能还在进行：暂停立刻生效，里程碑在后台做完
  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 2_000 });
  expect(await num(page, 'loss-current')).toBeLessThanOrEqual(initial * 0.85);

  await expect(empty).toBeHidden();
  expect(await pixels()).not.toBe(blank);
  // 停在 ≥ 200 步时，第 200 步的验证 loss 已经算出或正在算
  await expect.poll(() => num(page, 'loss-val'), { timeout: 60_000 }).toBeLessThan(initial);
});

// loss 区就是装着 loss 曲线的那个区块，生成记录区装着 samples
const lossPanel = (page: Page) => page.locator('section', { has: page.getByTestId('loss-chart') });
const samplesPanel = (page: Page) => page.locator('section', { has: page.getByTestId('samples') });

test('窄屏单列', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
  const loss = (await lossPanel(page).boundingBox())!;
  const samples = (await samplesPanel(page).boundingBox())!;
  expect(samples.y).toBeGreaterThan(loss.y + loss.height);
});

test('宽屏并排', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const loss = (await lossPanel(page).boundingBox())!;
  const samples = (await samplesPanel(page).boundingBox())!;
  expect(samples.x).toBeGreaterThan(loss.x);
  expect(Math.abs(samples.y - loss.y)).toBeLessThan(40);
});

test('重置', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const btn = page.getByTestId('start-toggle');
  await btn.click();
  await waitForStep(page, 3, 120_000);
  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 60_000 });
  await expect(page.getByTestId('loss-empty')).toBeHidden();
  await expect(page.getByTestId('sample-item')).toHaveCount(1);

  let message = '';
  page.once('dialog', (dialog) => {
    message = dialog.message();
    return dialog.accept();
  });
  await page.getByTestId('reset').click();
  await expect(btn).toHaveText('开始训练');
  expect(message).toBe('确定重置？当前训练进度会清空，存档不受影响。');
  await expect(page.getByTestId('step')).toHaveText('0');
  await expect(page.getByTestId('loss-initial')).toHaveText('—');
  await expect(page.getByTestId('samples-empty')).toBeVisible();
  await expect(page.getByTestId('loss-empty')).toBeVisible();
  await expect(page.getByTestId('heatmap-step')).toHaveText('来自第 0 步的模型', { timeout: 30_000 });
  await expect(btn).toBeEnabled();
});
