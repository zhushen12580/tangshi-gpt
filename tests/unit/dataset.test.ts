import { readFileSync } from 'node:fs';
import { expect, test } from 'vitest';
import { POEM_RE } from '../../scripts/build-dataset.mjs';
import presets from '../../src/presets.json';

const COMMIT = 'b8594f81a89752241442f2ce267d6f66f96704ee';
const data = JSON.parse(readFileSync('public/data/poems.json', 'utf8'));
const poems: string[] = data.poems;

test('每首诗都是格式严格的五绝，没有重复', () => {
  expect(poems.filter((p) => !POEM_RE.test(p))).toEqual([]);
  expect(new Set(poems).size).toBe(poems.length);
});

test('首数 ≥ 3000 且等于 count，commit 为固定值', () => {
  expect(poems.length).toBeGreaterThanOrEqual(3000);
  expect(data.count).toBe(poems.length);
  expect(data.commit).toBe(COMMIT);
  expect(data.license).toBe('MIT');
});

test('4 首预设诗都在数据集中，每个字都在字表中', () => {
  expect(presets.map((p) => p.title)).toEqual(['登鹳雀楼', '春晓', '相思', '鹿柴']);
  const chars = new Set(poems.join(''));
  for (const { poem } of presets) {
    expect(poems).toContain(poem);
    expect([...poem].filter((c) => !chars.has(c))).toEqual([]);
  }
});

test('NOTICE.txt 以 BOM 开头（服务器不带 charset 时浏览器也按 UTF-8 解码），含 commit 号和 MIT 许可全文', () => {
  const notice = readFileSync('public/data/NOTICE.txt', 'utf8');
  expect(notice.startsWith('﻿数据来源')).toBe(true);
  expect(notice).toContain(COMMIT);
  expect(notice).toContain('Permission is hereby granted');
});
