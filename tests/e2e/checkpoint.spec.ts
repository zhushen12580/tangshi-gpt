import { expect, test } from '@playwright/test';
import { num, openApp, waitForStep, waitReady } from './helpers';

test('保存、刷新、载入、继续', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const toggle = page.getByTestId('start-toggle');
  const status = page.getByTestId('save-status');
  const save = page.getByTestId('save');
  await expect(save).toBeEnabled();
  await toggle.click();
  await waitForStep(page, 100);
  // 此刻通常正在做第 100 步的里程碑（验证、写诗、热力图）；按钮和步数都要从点击起 2 秒内停下
  const clickedAt = Date.now();
  await toggle.click();
  await expect(toggle).toHaveText('继续', { timeout: 2_000 });
  expect(Date.now() - clickedAt).toBeLessThanOrEqual(2_000);
  await page.waitForTimeout(Math.max(0, clickedAt + 2_000 - Date.now()));
  const step = await num(page, 'step');
  await page.waitForTimeout(2_000);
  await expect(page.getByTestId('step')).toHaveText(String(step));

  await save.click();
  await expect(status).toHaveText(`已保存：第 ${step} 步`, { timeout: 60_000 });
  await page.reload();
  await waitReady(page);
  await expect(status).toContainText(`发现存档：第 ${step} 步`);
  const load = page.getByTestId('load');
  const ignore = page.getByTestId('ignore-save');
  await expect(load).toBeEnabled();
  await expect(ignore).toBeVisible();
  await ignore.click();
  await expect(ignore).toBeHidden();
  await expect(status).toHaveText('');
  await load.click();
  await expect(page.getByTestId('step')).toHaveText(String(step), { timeout: 60_000 });
  await expect(page.getByTestId('loss-empty')).toBeHidden();
  await expect.poll(() => page.getByTestId('loss-chart').evaluate((canvas: HTMLCanvasElement) =>
    canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data.some(value => value !== 0),
  )).toBe(true);
  expect(await page.getByTestId('sample-item').count()).toBeGreaterThanOrEqual(3);
  await expect(toggle).toHaveText('继续');
  await toggle.click();
  await waitForStep(page, step + 10);
  await toggle.click();
  await expect(toggle).toHaveText('继续', { timeout: 60_000 });
  expect(await num(page, 'loss-current')).toBeLessThanOrEqual(await num(page, 'loss-initial') * 0.85);

  page.once('dialog', dialog => dialog.accept());
  await page.getByTestId('delete-save').click();
  await expect(status).toHaveText('存档已删除');
  await expect(load).toBeDisabled();
  await page.reload();
  await waitReady(page);
  await expect(status).toHaveText('');
});

test('训练前保存的存档：载入后继续会写出第 0 步的诗，且只写一次', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const status = page.getByTestId('save-status');
  await page.getByTestId('save').click();
  await expect(status).toHaveText('已保存：第 0 步', { timeout: 60_000 });
  await page.reload();
  await waitReady(page);
  await page.getByTestId('load').click();
  await expect(status).toHaveText('已载入：第 0 步', { timeout: 60_000 });
  const toggle = page.getByTestId('start-toggle');
  await expect(toggle).toHaveText('继续');
  await toggle.click();
  const zero = page.getByTestId('sample-item')
    .filter({ has: page.locator('.sample-meta', { hasText: /^第 0 步 · 验证 loss \d+\.\d{3} · 自动/ }) });
  await expect(zero).toHaveCount(1, { timeout: 60_000 });
  await waitForStep(page, 1, 120_000);
  await toggle.click();
  await expect(toggle).toHaveText('继续', { timeout: 2_000 });
  await expect(zero).toHaveCount(1);
  await expect(page.getByTestId('loss-val')).toHaveText(/^\d+\.\d{3}$/);
});

test('IndexedDB 不可用时训练继续', async ({ page }) => {
  const errors: Error[] = [];
  page.on('pageerror', error => errors.push(error));
  await page.addInitScript(() => {
    Reflect.deleteProperty(window, 'indexedDB');
    Object.defineProperty(window, 'indexedDB', { value: undefined, configurable: true });
  });
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const save = page.getByTestId('save');
  await expect(save).toBeEnabled();
  await page.getByTestId('start-toggle').click();
  await waitForStep(page, 3);
  await save.click();
  await expect(page.getByTestId('save-status')).toHaveText('无法保存：此浏览器不支持本地存储', { timeout: 60_000 });
  await waitForStep(page, (await num(page, 'step')) + 1);
  expect(errors).toEqual([]);
});

test('不兼容的存档只能删除', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  await page.evaluate(() => new Promise<void>((resolve, reject) => {
    const request = indexedDB.open('tangshi-gpt', 1);
    request.onerror = () => reject(request.error);
    request.onupgradeneeded = () => request.result.createObjectStore('checkpoints');
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction('checkpoints', 'readwrite');
      tx.onabort = () => { db.close(); reject(tx.error); };
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.objectStore('checkpoints').put({
        version: 1, vocabHash: '00000000', config: {}, step: 5, weights: {}, optimizer: [],
        lossHistory: [], valHistory: [], samples: [], elapsedMs: 0, savedAt: Date.now(),
      }, 'latest');
    };
  }));
  await page.reload();
  await waitReady(page);
  const status = page.getByTestId('save-status');
  await expect(status).toHaveText('存档与当前数据集或模型不兼容');
  await expect(page.getByTestId('load')).toBeDisabled();
  await expect(page.getByTestId('save')).toBeDisabled();
  const remove = page.getByTestId('delete-save');
  await expect(remove).toBeEnabled();
  page.once('dialog', dialog => dialog.accept());
  await remove.click();
  await expect(status).toHaveText('存档已删除');
  await page.reload();
  await waitReady(page);
  await expect(status).toHaveText('');
});
