import { expect, test } from '@playwright/test';
import { holdDataset, num, openApp, waitForStep, waitReady } from './helpers';

test('首屏不加载视频', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', (request) => {
    if (request.url().includes('explainer.mp4')) requests.push(request.url());
  });
  await openApp(page);
  await waitReady(page);
  expect(requests).toEqual([]);
});

test('打开、播放、关闭', async ({ page }) => {
  await openApp(page);
  const open = page.getByTestId('explainer-open');
  const dialog = page.getByTestId('explainer-dialog');
  const video = page.getByTestId('explainer-video');
  await expect(open).toHaveText('1 分钟看懂原理');
  await open.click();
  await expect(dialog).toBeVisible();
  await expect(video).toHaveJSProperty('controls', true);
  await expect(video).not.toHaveAttribute('autoplay');
  await page.waitForTimeout(2_000);
  await expect(video).toHaveJSProperty('paused', true);
  await video.evaluate((el: HTMLVideoElement) => el.play());
  await expect.poll(() => video.evaluate((el: HTMLVideoElement) => el.currentTime), { timeout: 10_000 })
    .toBeGreaterThan(1);
  const duration = await video.evaluate((el: HTMLVideoElement) => el.duration);
  expect(duration).toBeGreaterThanOrEqual(60);
  expect(duration).toBeLessThanOrEqual(90);
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(video).toHaveJSProperty('paused', true);
  await expect(open).toBeFocused();
  await open.press('Enter');
  await expect(dialog).toBeVisible();
  await expect(video).toHaveJSProperty('paused', true);
  await page.getByTestId('explainer-close').click();
  await expect(dialog).toBeHidden();
  await expect(open).toBeFocused();
});

test('文字版', async ({ page }) => {
  await openApp(page);
  await page.getByTestId('explainer-open').click({ timeout: 5_000 });
  const transcript = page.getByTestId('explainer-transcript');
  await expect(transcript).toHaveAttribute('href', 'explainer.txt');
  await transcript.click();
  await expect(page).toHaveURL(/\/tangshi-gpt\/explainer\.txt$/);
  await expect(page.locator('body')).toContainText('猜下一个字');
});

test('训练中播放不影响训练', async ({ page }) => {
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  const toggle = page.getByTestId('start-toggle');
  await toggle.click();
  await waitForStep(page, 5);
  const before = await num(page, 'step');
  await page.getByTestId('explainer-open').click({ timeout: 5_000 });
  const video = page.getByTestId('explainer-video');
  await video.evaluate((el: HTMLVideoElement) => el.play());
  await page.waitForTimeout(10_000);
  await expect.poll(() => video.evaluate((el: HTMLVideoElement) => el.currentTime)).toBeGreaterThan(1);
  await page.getByTestId('explainer-close').click();
  await expect(page.getByTestId('explainer-dialog')).toBeHidden();
  expect(await num(page, 'step')).toBeGreaterThan(before);
  await expect(toggle).toHaveText('暂停');
  await toggle.click();
});

test('视频加载失败', async ({ page }) => {
  await page.route('**/explainer.mp4', (route) => route.fulfill({ status: 404, body: '' }));
  await openApp(page);
  await page.getByTestId('explainer-open').click({ timeout: 5_000 });
  const error = page.getByTestId('explainer-error');
  await expect(error).toBeVisible();
  await expect(error).toHaveText('视频加载失败，可以先看文字版。');
  await expect(page.getByTestId('explainer-transcript')).toBeVisible();
});

test('窄屏加载数据时也能打开对话框', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  const release = await holdDataset(page);
  try {
    await openApp(page);
    await expect(page.getByTestId('start-toggle')).toBeDisabled();
    await page.getByTestId('explainer-open').click({ timeout: 5_000 });
    for (const testId of ['explainer-dialog', 'explainer-video']) {
      const box = await page.getByTestId(testId).boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(375);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
  } finally {
    release();
  }
});
