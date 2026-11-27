import { expect, test } from '@playwright/test';
import { num, openApp, waitForStep, waitReady } from './helpers';

test('生成记录从第 0 步开始', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  await expect(page.getByTestId('samples-empty')).toHaveText('开始训练后，每 50 步会在这里写一首诗');

  const items = page.getByTestId('sample-item');
  const btn = page.getByTestId('start-toggle');
  await btn.click();
  // 点了就出来（review 发现 F-732d1f26dafd：以前要等验证集和 24 种长度的着色器，软件 WebGL 下 16 秒）；验证 loss 随后补上
  const zero = items.filter({ has: page.locator('.sample-meta', { hasText: '第 0 步' }) });
  await expect(zero).toHaveCount(1, { timeout: 5_000 });
  await expect(page.getByTestId('samples-empty')).toHaveCount(0);
  await expect(zero.locator('.sample-meta')).toHaveText(/^第 0 步 · 验证 loss \d+\.\d{3} · 自动/, { timeout: 30_000 });

  await waitForStep(page, 100);
  // 此刻通常正在做第 100 步的里程碑：暂停立刻生效，这首诗在后台写完。软件 WebGL 下 24 次单样本前向约 10 秒
  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 2_000 });
  await expect(items.filter({ has: page.locator('.sample-meta', { hasText: /^第 100 步 · 验证 loss \d/ }) })).toHaveCount(1, {
    timeout: 60_000,
  });

  const metas = await items.locator('.sample-meta').allTextContents();
  console.log(metas.join('\n'));
  const steps = metas.map((m) => Number(m.match(/^第 (\d+) 步/)![1]));
  for (const s of [0, 50, 100]) expect(steps).toContain(s);
  expect(steps[0]).toBe(Math.max(...steps)); // 最新的在最上面
  for (const m of metas) expect(m).toMatch(/验证 loss \d+\.\d{3}/);

  for (const item of await items.all()) {
    const lines = await item.locator('.poem-line').allTextContents();
    expect(lines).toHaveLength(4);
    for (const line of lines) expect(line).toMatch(/^[一-鿿]{5}[，。]$/);
    expect(lines.map((l) => l.at(-1)).join('')).toBe('，。，。');
  }
});

test('采样控件与键盘', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);

  // Tab 顺序：从页面开头连续按 Tab，控件按顺序出现，焦点框可见
  const tabbed: string[] = [];
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press('Tab');
    const f = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement;
      return { id: el.dataset.testid ?? el.tagName, outline: getComputedStyle(el).outlineStyle };
    });
    if (f.id === 'BODY') break;
    tabbed.push(f.id);
    expect(f.outline, f.id).not.toBe('none');
  }
  const order = ['start-toggle', 'reset', 'temperature', 'constrain', 'generate', 'save'].map((id) => tabbed.indexOf(id));
  expect(order.every((i) => i >= 0), tabbed.join(',')).toBe(true);
  expect(order).toEqual([...order].sort((a, b) => a - b));

  const slider = page.getByTestId('temperature');
  const items = page.getByTestId('sample-item');
  const metas = items.locator('.sample-meta');
  await expect(slider).toHaveValue('0.8');
  await expect(page.getByTestId('temperature-value')).toHaveText('0.8');

  // 训练前（第 0 步）也能立即生成。软件 WebGL 下第一次采样要编译着色器，十几秒
  await page.getByTestId('generate').click();
  await expect(metas.first()).toContainText('手动 · 温度 0.8 · 约束', { timeout: 60_000 });
  await expect(metas.first()).toContainText('第 0 步 · 验证 loss —');

  // 纯键盘：方向键调温度，空格切换复选框，Enter 触发按钮
  await slider.focus();
  for (let i = 0; i < 7; i++) await page.keyboard.press('ArrowRight');
  await expect(slider).toHaveValue('1.5');
  await expect(page.getByTestId('temperature-value')).toHaveText('1.5');
  await page.getByTestId('constrain').focus();
  await page.keyboard.press('Space');
  await expect(page.getByTestId('constrain')).not.toBeChecked();
  await page.getByTestId('generate').focus();
  await page.keyboard.press('Enter');
  await expect(items).toHaveCount(2, { timeout: 60_000 });
  await expect(metas.first()).toContainText('温度 1.5 · 自由');
  await expect(items.first().locator('.format-badge')).toHaveText(/^格式 [✓✗]$/);

  // 训练中立即生成：排到当前这一步之后，训练不中断
  const btn = page.getByTestId('start-toggle');
  await btn.click();
  await waitForStep(page, 1, 120_000); // 先等第 0 步的里程碑（写诗）做完
  await expect(metas.first()).toContainText('自动 · 温度 1.5 · 自由');
  const manual = items.filter({ hasText: '手动' });
  await page.getByTestId('generate').click();
  await expect(manual).toHaveCount(3, { timeout: 60_000 });
  await expect(manual.first().locator('.sample-meta')).toHaveText(
    /^第 [1-9]\d* 步 · 验证 loss \d+\.\d{3} · 手动 · 温度 1\.5 · 自由$/,
  );
  const stepAfter = await num(page, 'step');
  await waitForStep(page, stepAfter + 1, 60_000);
  await expect(btn).toHaveText('暂停');
  console.log((await metas.allTextContents()).join('\n'));

  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 60_000 });
});

test('排队中的「立即生成」按点击时的温度和格式开关写', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  // 在同一个任务里：0.8、约束时点一次，调到 1.5、取消约束再点一次。两次都排在队里，
  // 以前排到时才读控件，两条都成了「1.5 · 自由」（verify 发现 F-0f8a9dbb12ed）
  await page.evaluate(() => {
    const get = (id: string) => document.querySelector<HTMLInputElement>(`[data-testid="${id}"]`)!;
    get('generate').click();
    get('temperature').value = '1.5';
    get('temperature').dispatchEvent(new Event('input', { bubbles: true }));
    get('constrain').click();
    get('generate').click();
  });
  const metas = page.getByTestId('sample-item').locator('.sample-meta');
  await expect(metas).toHaveCount(2, { timeout: 60_000 });
  await expect(metas.nth(1)).toContainText('手动 · 温度 0.8 · 约束'); // 先点的在下面
  await expect(metas.nth(0)).toContainText('手动 · 温度 1.5 · 自由');
});
