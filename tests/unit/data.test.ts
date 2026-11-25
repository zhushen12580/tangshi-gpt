import { readFileSync } from 'node:fs';
import { expect, test } from 'vitest';
import {
  BOS,
  buildVocab,
  createDataset,
  decode,
  encode,
  loadDataset,
  makeBatch,
  splitTrainVal,
  vocabHash,
  type PoemsFile,
} from '../../src/data';
import { mulberry32 } from './rng';

const file: PoemsFile = JSON.parse(readFileSync('public/data/poems.json', 'utf8'));

test('buildVocab 去重、按码点升序，最后一个是 <s>', () => {
  const vocab = buildVocab(['乙甲，。', '甲丙，。']);
  expect(vocab).toEqual(['。', '丙', '乙', '甲', '，', BOS]);
  expect(BOS).toBe('<s>');
});

test('encode / decode 往返一致', () => {
  const vocab = buildVocab(file.poems.slice(0, 10));
  const stoi = new Map(vocab.map((c, i) => [c, i]));
  const text = file.poems[3];
  const ids = encode(text, stoi);
  expect(ids).toHaveLength(24);
  expect(decode(ids, vocab)).toBe(text);
});

test('splitTrainVal：下标 % 20 == 0 的进验证集', () => {
  const items = Array.from({ length: 41 }, (_, i) => i);
  const { train, val } = splitTrainVal(items);
  expect(val).toEqual([0, 20, 40]);
  expect(train).toHaveLength(38);
});

test('makeBatch：输入是 <s> 加目标右移一位', () => {
  const seqs = [
    Array.from({ length: 24 }, (_, i) => i),
    Array.from({ length: 24 }, (_, i) => 100 + i),
  ];
  const bos = 999;
  const { inputs, targets } = makeBatch(seqs, bos, 3, mulberry32(1));
  expect(inputs).toBeInstanceOf(Int32Array);
  expect(inputs).toHaveLength(3 * 24);
  expect(targets).toHaveLength(3 * 24);
  for (let r = 0; r < 3; r++) {
    expect(inputs[r * 24]).toBe(bos);
    for (let k = 1; k < 24; k++) expect(inputs[r * 24 + k]).toBe(targets[r * 24 + k - 1]);
    expect(seqs.map((s) => s.join())).toContain(Array.from(targets.slice(r * 24, r * 24 + 24)).join());
  }
});

test('vocabHash 是确定的 8 位十六进制，字表变了结果就变', () => {
  const h = vocabHash(['甲', '乙', BOS]);
  expect(h).toMatch(/^[0-9a-f]{8}$/);
  expect(vocabHash(['甲', '乙', BOS])).toBe(h);
  expect(vocabHash(['甲', '丙', BOS])).not.toBe(h);
});

test('createDataset 处理真实数据集', () => {
  const ds = createDataset(file);
  expect(ds.bosId).toBe(ds.vocab.length - 1);
  expect(ds.vocab[ds.bosId]).toBe(BOS);
  expect(ds.train.length + ds.val.length).toBe(file.count);
  expect(ds.train[0]).toHaveLength(24);
  expect(ds.hash).toBe(vocabHash(ds.vocab));
});

test('loadDataset 成功时返回 Dataset', async () => {
  const fakeFetch = (async () => new Response(JSON.stringify(file))) as typeof fetch;
  const ds = await loadDataset('data/poems.json', fakeFetch);
  expect(ds.poems).toHaveLength(file.count);
});

test('loadDataset 收到 404 时 reject，错误信息含 404', async () => {
  const fakeFetch = (async () => new Response('not found', { status: 404 })) as typeof fetch;
  await expect(loadDataset('data/poems.json', fakeFetch)).rejects.toThrow('404');
});
