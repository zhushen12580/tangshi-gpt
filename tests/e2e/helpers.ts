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

/** 扣住 poems.json，返回放行函数。比固定推迟确定：机器忙时也一定能看到加载中的状态，10 秒的检查从放行算起 */
export async function holdDataset(page: Page): Promise<() => void> {
  let release!: () => void;
  const released = new Promise<void>((resolve) => (release = resolve));
  await page.route('**/data/poems.json', async (r) => {
    await released;
    await r.continue();
  });
  return release;
}

/** 第一张热力图画好了：它排在「开始训练」可用之后才画，软件 WebGL 下要编译一批着色器 */
export async function waitHeatmap(page: Page): Promise<void> {
  await expect(page.getByTestId('heatmap-step')).toHaveText('来自第 0 步的模型', { timeout: 30_000 });
}

export async function num(page: Page, testId: string): Promise<number> {
  return parseFloat((await page.getByTestId(testId).textContent()) ?? '');
}

/** 等到步数 ≥ n（软件 WebGL 下 200 步要几分钟） */
export async function waitForStep(page: Page, n: number, timeout = 540_000): Promise<void> {
  await expect.poll(() => num(page, 'step'), { timeout }).toBeGreaterThanOrEqual(n);
}
