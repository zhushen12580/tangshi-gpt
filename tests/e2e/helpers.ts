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

/** 等到步数 ≥ n（软件 WebGL 下 200 步要几分钟） */
export async function waitForStep(page: Page, n: number, timeout = 540_000): Promise<void> {
  await expect.poll(() => num(page, 'step'), { timeout }).toBeGreaterThanOrEqual(n);
}
