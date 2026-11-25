import { expect, test } from '@playwright/test';
import { openApp, waitForStep, waitReady } from './helpers';

test('生成记录从第 0 步开始', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  await expect(page.getByTestId('samples-empty')).toHaveText('开始训练后，每 50 步会在这里写一首诗');

  const items = page.getByTestId('sample-item');
  const btn = page.getByTestId('start-toggle');
  await btn.click();
  await expect(items.filter({ has: page.locator('.sample-meta', { hasText: '第 0 步' }) })).toHaveCount(1, {
    timeout: 30_000,
  });
  await expect(page.getByTestId('samples-empty')).toHaveCount(0);

  await waitForStep(page, 100);
  // 此刻通常正在做第 100 步的里程碑：暂停要等这首诗写完。软件 WebGL 下 24 次单样本前向约 10 秒
  await btn.click();
  await expect(btn).toHaveText('继续', { timeout: 60_000 });

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
