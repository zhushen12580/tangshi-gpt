import { expect, test } from '@playwright/test';
import presets from '../../src/presets.json' with { type: 'json' };
import { openApp, waitForStep, waitReady } from './helpers';

const INTRO =
  '它只做一件事：根据前面的字，猜下一个字。一个字一个字猜满 24 个（20 个字加 4 个标点），就是一首五言绝句。训练就是让它反复去猜真唐诗、对答案，再把参数往猜对的方向挪一点。';

test('大白话与问号', async ({ page }) => {
  await openApp(page);
  await waitReady(page);
  await expect(page.getByTestId('explain-intro')).toHaveText(INTRO);

  const helps = page.locator('[data-testid^="help-"]:not([data-testid$="-text"])');
  await expect(helps).toHaveCount(11);
  for (const label of await helps.evaluateAll((els) => els.map((el) => el.getAttribute('aria-label') ?? ''))) {
    expect(label).toMatch(/^说明：/);
    expect(label).not.toContain('查看');
  }

  const baseline = page.getByTestId('help-baseline-text');
  await page.getByTestId('help-baseline').click();
  await expect(baseline).toBeVisible();
  await expect(baseline).toContainText('3605');
  await expect(baseline).toContainText('8.19');
  await page.keyboard.press('Escape');
  await expect(baseline).toBeHidden();

  const constrain = page.getByTestId('help-constrain-text');
  await page.getByTestId('help-constrain').click();
  for (const s of ['硬规则', '不是模型学会的', '关掉它']) await expect(constrain).toContainText(s);
  await page.getByTestId('help-loss-val').click();
  await expect(constrain).toBeHidden();
  const val = page.getByTestId('help-loss-val-text');
  await expect(val).toBeVisible();
  await expect(val).toContainText('模拟考');
  await expect(val).toContainText('184');
});

test('键盘', async ({ page }) => {
  await openApp(page);
  await waitReady(page);
  const btn = page.getByTestId('help-baseline');
  await btn.focus();
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Tab'); // 键盘聚焦，:focus-visible 才生效
  await expect(btn).toBeFocused();
  expect(await btn.evaluate((el) => getComputedStyle(el).outlineStyle)).not.toBe('none');
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('help-baseline-text')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('help-baseline-text')).toBeHidden();
  await expect(btn).toBeFocused();
});

test('当前 loss 的说明随训练更新', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const guess = page.getByTestId('loss-current-guess');
  await expect(guess).toHaveText('还没开始训练。');

  const toggle = page.getByTestId('start-toggle');
  await toggle.click();
  await expect(toggle).toHaveText('暂停');
  await page.getByTestId('help-speed').click();
  await expect(page.getByTestId('help-speed-text')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('help-speed-text')).toBeHidden();
  await expect(toggle).toHaveText('暂停');

  await waitForStep(page, 20);
  await toggle.click();
  await expect(toggle).toHaveText('继续', { timeout: 2_000 });
  // 两个元素在同一个函数里更新，一次读出来才不会碰上中间状态
  const [x, text] = await page.evaluate(() =>
    ['loss-current', 'loss-current-guess'].map((id) => document.querySelector(`[data-testid="${id}"]`)!.textContent),
  );
  expect(x).toMatch(/^\d+\.\d{3}$/);
  expect(text).toBe(`现在 ${x} ≈ 在 ${Math.max(1, Math.round(Math.exp(Number(x))))} 个字里瞎蒙。`);
});

test('数字来自数据', async ({ page }) => {
  await page.route('**/data/poems.json', (r) =>
    r.fulfill({ json: { source: 'test', commit: 'test', license: 'MIT', count: 4, poems: presets.map((p) => p.poem) } }),
  );
  await openApp(page);
  await waitReady(page);
  await page.getByTestId('help-baseline').click();
  await expect(page.getByTestId('help-baseline-text')).toContainText('74');
  await expect(page.getByTestId('help-baseline-text')).toContainText('4.30');
  await page.getByTestId('help-loss-val').click();
  await expect(page.getByTestId('help-loss-val-text')).toContainText('1 首诗');
});

test('窄屏', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await openApp(page);
  await waitReady(page);
  await page.getByTestId('help-heatmap').click();
  const card = page.getByTestId('help-heatmap-text');
  await expect(card).toBeVisible();
  const box = (await card.boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(375);
  expect(box.y + box.height).toBeLessThanOrEqual(667);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
});
