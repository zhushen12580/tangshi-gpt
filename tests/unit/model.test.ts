import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { beforeAll, expect, test } from 'vitest';
import {
  MODEL_DEFAULTS,
  countParams,
  forward,
  initParams,
  lossFn,
  nextTokenLogits,
  type GPTConfig,
} from '../../src/model';
import { mulberry32 } from './rng';

const MICRO: GPTConfig = { vocabSize: 11, ctx: 8, nLayer: 2, nHead: 2, dModel: 16, dFF: 32 };

/** 用 mulberry32 生成 [rows, cols] 的随机 id */
function randomIds(rows: number, cols: number, vocab: number, seed: number): tf.Tensor2D {
  const rng = mulberry32(seed);
  const data = Array.from({ length: rows * cols }, () => Math.floor(rng() * vocab));
  return tf.tensor2d(data, [rows, cols], 'int32');
}

beforeAll(async () => {
  await tf.setBackend('cpu');
});

test('参数量：微型配置 4784，默认配置 V=3500 时 1046144', () => {
  expect(countParams(initParams(MICRO, 1))).toBe(4784);
  const n = countParams(initParams({ ...MODEL_DEFAULTS, vocabSize: 3500 }, 1));
  expect(n).toBe(1046144);
  expect(n).toBeGreaterThanOrEqual(0.8e6);
  expect(n).toBeLessThanOrEqual(1.3e6);
});

test('forward 输出形状正确，注意力每行和为 1，右上三角为 0', () => {
  const p = initParams(MICRO, 1);
  const { logits, attn } = forward(p, MICRO, randomIds(2, 8, 11, 7));
  expect(logits.shape).toEqual([2, 8, 11]);
  expect(attn.length).toBe(2);
  for (const a of attn) {
    expect(a.shape).toEqual([2, 2, 8, 8]);
    const rows = a.arraySync();
    for (const b of rows)
      for (const h of b)
        h.forEach((row, i) => {
          expect(Math.abs(row.reduce((s, x) => s + x, 0) - 1)).toBeLessThan(1e-4);
          row.forEach((x, j) => {
            if (j > i) expect(x).toBeLessThan(1e-6);
          });
        });
  }
});

test('因果性：只改第 5 位输入，位置 0–4 的 logits 不变', () => {
  const p = initParams(MICRO, 2);
  const a = [1, 2, 3, 4, 5, 6, 7, 8];
  const b = [...a];
  b[5] = 0;
  const la = forward(p, MICRO, tf.tensor2d([a], [1, 8], 'int32')).logits.arraySync()[0];
  const lb = forward(p, MICRO, tf.tensor2d([b], [1, 8], 'int32')).logits.arraySync()[0];
  for (let t = 0; t < 5; t++)
    la[t].forEach((x, v) => expect(Math.abs(x - lb[t][v])).toBeLessThan(1e-5));
  // 第 5 位本身应当变化，否则上面的断言没有意义
  expect(la[5].some((x, v) => Math.abs(x - lb[5][v]) > 1e-5)).toBe(true);
});

test('默认配置 V=3500 的初始 loss 接近 ln(3500)', () => {
  const cfg = { ...MODEL_DEFAULTS, vocabSize: 3500 };
  const p = initParams(cfg, 3);
  const loss = lossFn(p, cfg, randomIds(4, 24, 3500, 11), randomIds(4, 24, 3500, 12));
  expect(loss.shape).toEqual([]);
  expect(Math.abs(loss.dataSync()[0] - Math.log(3500))).toBeLessThan(0.2);
});

test('可学习：微型配置固定 4 条序列，adam 1e-2 训练 100 步后 loss 减半', () => {
  const p = initParams(MICRO, 4);
  const inputs = randomIds(4, 8, 11, 21);
  const targets = randomIds(4, 8, 11, 22);
  const opt = tf.train.adam(1e-2);
  const varList = Object.values(p);
  const initial = lossFn(p, MICRO, inputs, targets).dataSync()[0];
  let last = initial;
  for (let i = 0; i < 100; i++) {
    const loss = tf.tidy(() => opt.minimize(() => lossFn(p, MICRO, inputs, targets), true, varList)!);
    last = loss.dataSync()[0];
    loss.dispose();
  }
  expect(last).toBeLessThan(initial * 0.5);
});

test('同一 seed 两次初始化，wte 完全相同', () => {
  const a = initParams(MICRO, 5).wte.dataSync();
  const b = initParams(MICRO, 5).wte.dataSync();
  expect(Array.from(a)).toEqual(Array.from(b));
});

test('nextTokenLogits 等于 forward 在最后位置的 logits', async () => {
  const p = initParams(MICRO, 6);
  const got = await nextTokenLogits(p, MICRO, [1, 2, 3]);
  const want = forward(p, MICRO, tf.tensor2d([[1, 2, 3]], [1, 3], 'int32')).logits.arraySync()[0][2];
  expect(got).toHaveLength(11);
  Array.from(got).forEach((x, v) => expect(Math.abs(x - want[v])).toBeLessThan(1e-5));
});
