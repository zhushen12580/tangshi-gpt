import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // 必须限定 include，否则 tests/e2e/*.spec.ts 也会被当成单测
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
    // 模型训练类单测正常约 2.5 秒；覆盖率插桩加并行负载下会超过默认的 5 秒
    testTimeout: 20_000,
    coverage: {
      provider: 'v8',
      include: ['src/data.ts', 'src/model.ts', 'src/sample.ts', 'src/train.ts', 'src/checkpoint.ts'],
      thresholds: { lines: 80 },
    },
  },
});
