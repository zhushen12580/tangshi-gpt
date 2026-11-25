import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // 必须限定 include，否则 tests/e2e/*.spec.ts 也会被当成单测
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
  },
});
