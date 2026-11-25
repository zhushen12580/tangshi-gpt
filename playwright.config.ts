import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  workers: 1, // 避免多个 GPU 训练互相争抢
  timeout: 600_000,
  use: {
    baseURL: 'http://localhost:4173/tangshi-gpt/',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // 没有 GPU 的无人值守环境下也有软件 WebGL
        launchOptions: { args: ['--enable-unsafe-swiftshader'] },
      },
    },
  ],
  webServer: {
    command: 'npm run build && npm run preview',
    url: 'http://localhost:4173/tangshi-gpt/',
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
