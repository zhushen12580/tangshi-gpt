import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import * as OpenCC from 'opencc-js';
import { describe, expect, test } from 'vitest';
import { selectPoems } from '../../scripts/build-dataset.mjs';

const t2cn = OpenCC.Converter({ from: 't', to: 'cn' });
const poem = (...paragraphs: string[]) => ({ paragraphs });

describe('selectPoems', () => {
  test('繁体合格五绝转成简体', () => {
    const files = [[poem('白日依山盡，黃河入海流。', '欲窮千里目，更上一層樓。')]];
    expect(selectPoems(files, t2cn)).toEqual(['白日依山尽，黄河入海流。欲穷千里目，更上一层楼。']);
  });

  test('过滤七言、带「？」、三句的诗', () => {
    const files = [
      [
        poem('朝辭白帝彩雲間，千里江陵一日還。', '兩岸猿聲啼不住，輕舟已過萬重山。'),
        poem('君自故鄉來，應知故鄉事？', '來日綺窗前，寒梅著花未？'),
        poem('春眠不覺曉，處處聞啼鳥。', '夜來風雨聲，'),
      ],
    ];
    expect(selectPoems(files, t2cn)).toEqual([]);
  });

  test('转简体后相同的诗只保留第一首，并保持源顺序', () => {
    const files = [
      [poem('紅豆生南國，春來發幾枝。', '願君多采擷，此物最相思。')],
      [
        poem('白日依山盡，黃河入海流。', '欲窮千里目，更上一層樓。'),
        poem('白日依山尽，黄河入海流。', '欲穷千里目，更上一层楼。'),
      ],
      [poem('空山不見人，但聞人語響。', '返景入深林，復照青苔上。')],
    ];
    expect(selectPoems(files, t2cn)).toEqual([
      '红豆生南国，春来发几枝。愿君多采撷，此物最相思。',
      '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。',
      '空山不见人，但闻人语响。返景入深林，复照青苔上。',
    ]);
  });
});

test('下载失败时退出码非 0、报出文件名，且不改动 poems.json', () => {
  const file = 'public/data/poems.json';
  const before = existsSync(file) ? readFileSync(file) : null;
  const r = spawnSync(process.execPath, ['scripts/build-dataset.mjs'], {
    env: { ...process.env, CHINESE_POETRY_BASE: 'http://127.0.0.1:9/' },
    encoding: 'utf8',
    timeout: 60_000,
  });
  expect(r.status).not.toBe(0);
  expect(r.stderr).toContain('poet.tang.');
  if (before === null) expect(existsSync(file)).toBe(false);
  else expect(readFileSync(file).equals(before)).toBe(true);
}, 70_000);
