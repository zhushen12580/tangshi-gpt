import { expect, type Page } from '@playwright/test';

export async function openApp(page: Page, query = ''): Promise<void> {
  await page.goto('./' + query);
}

/** 「开始训练」在 10 秒内可用 */
export async function waitReady(page: Page): Promise<void> {
  const btn = page.getByTestId('start-toggle');
  await expect(btn).toBeEnabled({ timeout: 10_000 });
  await expect(btn).toHaveText('开始训练');
}

export async function num(page: Page, testId: string): Promise<number> {
  return parseFloat((await page.getByTestId(testId).textContent()) ?? '');
}
