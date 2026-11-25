import { expect, test } from '@playwright/test';

test('首页在 /tangshi-gpt/ 子路径下显示标题', async ({ page }) => {
  await page.goto('./');
  await expect(page.getByRole('heading', { name: '唐诗 GPT' })).toBeVisible();
});
