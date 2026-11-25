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

test('初始验证 loss 接近 ln(V)，且不留下张量', async () => {
  const t = make();
  const before = tf.memory().numTensors;
  const v = await t.evalValLoss();
  expect(Math.abs(v - Math.log(11))).toBeLessThan(0.3);
  expect(tf.memory().numTensors).toBe(before);
});

test('第 0 步和每 50 步做里程碑：先算验证 loss 再回调；停在第 50 步时它的里程碑也已完成', async () => {
  const seen: { step: number; valLoss: number; trained: number }[] = [];
  const t = make({
    onMilestone: (m) => void seen.push({ ...m, trained: t.lossHistory.length }),
    onStep: (s) => void (s.step === 50 && t.pause()),
  });
  t.start();
  await vi.waitFor(() => expect(t.state).toBe('paused'), { timeout: 20_000 });
  expect(t.stats().step).toBe(50);
  expect(t.valHistory.map((v) => v.step)).toEqual([0, 50]);
  expect(seen.map((m) => m.step)).toEqual([0, 50]);
  expect(seen[0].trained).toBe(0); // 第 0 步的里程碑在第一步训练之前
  expect(seen.map((m) => m.valLoss)).toEqual(t.valHistory.map((v) => v.loss));
  expect(t.valHistory[1].loss).toBeLessThan(t.valHistory[0].loss);

  // 继续训练时不会重复第 50 步的里程碑
  t.start();
  await untilStep(52);
  await t.pause();
  expect(t.valHistory.map((v) => v.step)).toEqual([0, 50]);
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
