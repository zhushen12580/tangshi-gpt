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

test('仓库地址：README 的 clone 命令和页脚 GitHub 链接指向同一个公开仓库', () => {
  const REPO = 'https://github.com/zhushen12580/tangshi-gpt';
  const readme = read('README.md');
  const html = read('index.html');
  expect(readme).toContain(`git clone ${REPO}.git`);
  expect(html).toContain(`<a href="${REPO}">GitHub</a>`);
  // fork、少了 .git、大小写不同都会让陌生人遇到 404；区分大小写比较
  const urls = (readme + html).match(/https:\/\/github\.com\/zhushen12580\/[\w.-]*/gi) ?? [];
  expect(urls.length).toBeGreaterThanOrEqual(2);
  for (const url of urls) expect([REPO, `${REPO}.git`]).toContain(url);
});

test('在线体验地址：两份 README 都链接到本仓库的 Pages 站点，GitHub 渲染后链接不多带字', () => {
  // Pages 地址是 https://<owner>.github.io/<repo>/，与 vite 子路径 /tangshi-gpt/ 一致；站点是否已启用只能联网查，见 progress。
  // GFM 的裸网址一直延伸到空白或 <，只去掉结尾的 ?!.,:*_~，所以「网址**（推荐」会把「**（推荐」也算进链接
  const PAGES = 'https://zhushen12580.github.io/tangshi-gpt/';
  for (const path of ['README.md', 'README.zh-CN.md']) {
    const prose = read(path).replace(/```[\s\S]*?```/g, '').replace(/`[^`]*`/g, '');
    const bare = [...prose.matchAll(/(?<![(<])https?:\/\/[^\s<]*/g)].map(([u]) => u.replace(/[?!.,:*_~]+$/, ''));
    for (const url of bare) expect(url, path).toMatch(/^[\x21-\x29\x2B-\x7E]+$/); // 只有可见 ASCII，且没有 *
    const pages = prose.match(/https:\/\/[\w-]+\.github\.io\/[^\s)*>]*/g) ?? [];
    expect(pages, path).toContain(PAGES);
    for (const url of pages) expect(url, path).toBe(PAGES);
  }
});
