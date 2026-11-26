import { readFileSync } from 'node:fs';
import { expect, test } from 'vitest';

const read = (path: string) => readFileSync(path, 'utf8');
const engines = (path: string): string => JSON.parse(read(path)).engines.node;

test('Pages 工作流：安装依赖、跑单测、构建，然后发布 dist/', () => {
  const yml = read('.github/workflows/pages.yml');
  const order = ['npm ci', 'npm test', 'npm run build', 'actions/upload-pages-artifact', 'actions/deploy-pages'];
  const at = order.map((s) => yml.indexOf(s));
  expect(at.every((i) => i >= 0)).toBe(true);
  expect(at).toEqual([...at].sort((a, b) => a - b));
  expect(yml).toMatch(/path: dist\b/);
  expect(yml).toMatch(/branches: \[main\]/);
});

test('.env.example 写明 build:data 的可选变量 CHINESE_POETRY_BASE', () => {
  expect(read('.env.example')).toMatch(/^CHINESE_POETRY_BASE=https:\/\//m);
});

test('engines.node 与锁定的 Vitest 一致（它的范围最窄，且落在 Vite 的范围内）', () => {
  // vite 8：^20.19.0 || >=22.12.0；vitest 5：^22.12.0 || ^24.0.0 || >=26.0.0。工具升级后范围变了，这里会提醒
  expect(engines('node_modules/vite/package.json')).toBe('^20.19.0 || >=22.12.0');
  expect(engines('package.json')).toBe(engines('node_modules/vitest/package.json'));
  expect(JSON.parse(read('package-lock.json')).packages[''].engines.node).toBe(engines('package.json'));
});
