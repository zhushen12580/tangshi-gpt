import { readdirSync } from 'node:fs';
import { extname } from 'node:path';
import { expect, test } from '@playwright/test';
import { openApp, waitForStep, waitReady } from './helpers';

const STATIC_EXTS = ['.html', '.js', '.css', '.json', '.txt'];

test('dist 只有静态文件', () => {
  // webServer 启动前已经执行过 npm run build
  const dist = new URL('../../dist/', import.meta.url);
  const files = readdirSync(dist, { recursive: true, withFileTypes: true })
    .filter((d) => d.isFile())
    .map((d) => `${d.parentPath}/${d.name}`.replaceAll('\\', '/'));
  expect(files.length).toBeGreaterThan(0);
  expect(files.filter((f) => !STATIC_EXTS.includes(extname(f)))).toEqual([]);
  expect(files.some((f) => f.endsWith('/dist/data/poems.json'))).toBe(true);
  expect(files.some((f) => f.endsWith('/dist/data/NOTICE.txt'))).toBe(true);
});

test('子路径下没有 404', async ({ page }) => {
  const failed: string[] = [];
  page.on('response', (r) => {
    if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`);
  });
  await openApp(page, '?backend=webgl');
  await waitReady(page);
  await expect(page.getByTestId('dataset-info')).toBeVisible();

  const btn = page.getByTestId('start-toggle');
  await btn.click();
  await waitForStep(page, 5);
  await btn.click();
  await expect(btn).toHaveText('继续');
  expect(failed).toEqual([]);
});

test('页脚「许可说明」的中文不乱码', async ({ page }) => {
  // 服务器返回的 text/plain 不带 charset，Chromium 会按 Windows-1252 解码，只能靠文件开头的 BOM 声明 UTF-8
  await openApp(page);
  await page.getByRole('link', { name: '许可说明' }).click();
  await expect(page).toHaveURL(/\/tangshi-gpt\/data\/NOTICE\.txt$/);
  await expect(page.locator('body')).toContainText('数据来源：chinese-poetry/chinese-poetry');
  await expect(page.locator('body')).toContainText('Permission is hereby granted');
});
