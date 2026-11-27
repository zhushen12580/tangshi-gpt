// 从 chinese-poetry 的固定 commit 重新生成 public/data/poems.json 和 NOTICE.txt。
// 用法：npm run build:data
// CHINESE_POETRY_BASE 可改成镜像 URL（http/https 开头）或本地 clone 目录。
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as OpenCC from 'opencc-js';

const COMMIT = 'b8594f81a89752241442f2ce267d6f66f96704ee';
const DEFAULT_BASE = `https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/${COMMIT}/`;
const POEM_FILES = Array.from({ length: 58 }, (_, i) => `全唐诗/poet.tang.${i * 1000}.json`);
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const OUT_DIR = join(ROOT, 'public', 'data');

export const POEM_RE = /^([一-鿿]{5}，[一-鿿]{5}。){2}$/u;

/**
 * 拼接 paragraphs → 严格五绝 → 繁转简 → 再校验 → 去重，保持源顺序。
 * @param {Array<Array<{paragraphs: string[]}>>} files
 * @param {(s: string) => string} convert
 * @returns {string[]}
 */
export function selectPoems(files, convert) {
  const poems = new Set();
  for (const file of files) {
    for (const { paragraphs } of file) {
      const raw = Array.isArray(paragraphs) ? paragraphs.join('') : '';
      if (!POEM_RE.test(raw)) continue;
      const poem = convert(raw);
      if (POEM_RE.test(poem)) poems.add(poem);
    }
  }
  return [...poems];
}

async function readSource(base, path) {
  if (!/^https?:\/\//.test(base)) return readFile(join(base, path), 'utf8');
  const url = new URL(path.split('/').map(encodeURIComponent).join('/'), base.endsWith('/') ? base : `${base}/`);
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(30_000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch (err) {
      lastError = err;
      if (attempt < 3) await new Promise((r) => setTimeout(r, 500));
    }
  }
  throw lastError;
}

function parsePoemFile(text) {
  const value = JSON.parse(text);
  if (!Array.isArray(value)) throw new Error('不是 JSON 数组');
  return value;
}

async function main() {
  const base = process.env.CHINESE_POETRY_BASE || DEFAULT_BASE;
  const paths = [...POEM_FILES, 'LICENSE'];
  const results = await Promise.allSettled(
    paths.map((p) => readSource(base, p).then((text) => (p === 'LICENSE' ? text : parsePoemFile(text)))),
  );
  const failures = results.flatMap((r, i) =>
    r.status === 'rejected' ? [`下载失败：${paths[i]}（${r.reason?.cause?.message ?? r.reason?.message ?? r.reason}）`] : [],
  );
  if (failures.length > 0) throw new Error(failures.join('\n'));

  const files = results.slice(0, -1).map((r) => r.value);
  const license = results.at(-1).value.replace(/\r\n/g, '\n').trimEnd();
  const poems = selectPoems(files, OpenCC.Converter({ from: 't', to: 'cn' }));

  const presetsFile = join(ROOT, 'src', 'presets.json');
  if (existsSync(presetsFile)) {
    const missing = JSON.parse(await readFile(presetsFile, 'utf8')).filter(({ poem }) => !poems.includes(poem));
    if (missing.length > 0) throw new Error(`预设诗不在数据集中：${missing.map((p) => p.title).join('、')}`);
  }

  const data = { source: 'chinese-poetry/chinese-poetry 全唐诗', commit: COMMIT, license: 'MIT', count: poems.length, poems };
  // 开头的 BOM 让浏览器按 UTF-8 解码：很多静态服务器（含 vite dev/preview）返回的 text/plain 不带 charset
  const notice =
    `﻿数据来源：chinese-poetry/chinese-poetry（https://github.com/chinese-poetry/chinese-poetry），commit ${COMMIT}，` +
    `经 opencc-js 繁转简，仅保留格式严格的五言绝句。\n\n${license}\n`;
  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(join(OUT_DIR, 'poems.json'), `${JSON.stringify(data)}\n`);
  await writeFile(join(OUT_DIR, 'NOTICE.txt'), notice);
  console.log(`写入 ${poems.length} 首`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err.message);
    process.exit(1);
  });
}
