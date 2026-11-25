import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { afterEach, beforeAll, expect, test, vi } from 'vitest';
import type { GPTConfig } from '../../src/model';
import { Trainer, type TrainerOptions } from '../../src/train';
import { mulberry32 } from './rng';

const MICRO: GPTConfig = { vocabSize: 11, ctx: 8, nLayer: 2, nHead: 2, dModel: 16, dFF: 32 };
const BOS = 10;

// 40 条有规律的序列：随机起点，之后每个字 +1（模 10），id 0–9，<s> = 10
const rng = mulberry32(42);
const SEQS = Array.from({ length: 40 }, () => {
  const s = Math.floor(rng() * 10);
  return Array.from({ length: 8 }, (_, k) => (s + k) % 10);
});

let trainer: Trainer;
const make = (extra: Partial<TrainerOptions> = {}) =>
  (trainer = new Trainer({
    cfg: MICRO,
    train: SEQS,
    val: SEQS.slice(0, 4),
    bosId: BOS,
    batchSize: 8,
    lr: 1e-2,
    seed: 1,
    rng: mulberry32(1),
    ...extra,
  }));

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const untilStep = (n: number) => vi.waitFor(() => expect(trainer.stats().step).toBeGreaterThanOrEqual(n), { timeout: 10_000 });

beforeAll(async () => {
  await tf.setBackend('cpu');
});

afterEach(async () => {
  await trainer.pause();
  trainer.dispose();
});

test('step() 记录 loss 并更新统计，loss 下降', async () => {
  const t = make();
  for (let i = 0; i < 30; i++) await t.step();
  const h = t.lossHistory;
  expect(h.length).toBe(30);
  const s = t.stats();
  expect(s.step).toBe(30);
  expect(s.initialLoss).toBe(h[0]);
  expect(s.currentLoss).toBeCloseTo(h.slice(-10).reduce((a, b) => a + b, 0) / 10, 10);
  expect(h[29]).toBeLessThan(h[0]);
  expect(t.state).toBe('idle');
});

test('连续训练时张量数量不增长', async () => {
  const t = make();
  await t.step();
  await t.step(); // 热身：Adam 在第一步创建一阶、二阶矩变量
  const before = tf.memory().numTensors;
  for (let i = 0; i < 20; i++) await t.step();
  expect(tf.memory().numTensors).toBe(before);
});

test('start() 后 pause() 等当前步结束才返回，暂停期间步数和用时不变', async () => {
  const states: string[] = [];
  const t = make({ onState: (s) => states.push(s) });
  t.start();
  expect(t.state).toBe('running');
  await untilStep(3);
  await t.pause();
  expect(t.state).toBe('paused');
  const { step, elapsedMs } = t.stats();
  expect(elapsedMs).toBeGreaterThan(0);
  await sleep(50);
  expect(t.stats().step).toBe(step);
  expect(t.stats().elapsedMs).toBe(elapsedMs);
  expect(t.stats().stepsPerSec).toBeGreaterThan(0);
  expect(states).toEqual(['running', 'paused']);
});

test('连调两次 start() 也只有一个训练循环', async () => {
  const t = make();
  const step = t.step.bind(t);
  let active = 0;
  let maxActive = 0;
  vi.spyOn(t, 'step').mockImplementation(async () => {
    maxActive = Math.max(maxActive, ++active);
    try {
      return await step();
    } finally {
      active--;
    }
  });
  t.start();
  t.start();
  await untilStep(5);
  await t.pause();
  expect(maxActive).toBe(1);
});
