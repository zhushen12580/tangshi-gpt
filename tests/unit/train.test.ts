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

test('恢复权重、历史和 Adam，使用相同 batch 继续训练时结果等价', async () => {
  let random = mulberry32(1);
  const a = make({ rng: () => random() });
  const b = new Trainer({ cfg: MICRO, train: SEQS, val: SEQS.slice(0, 4), bosId: BOS,
    batchSize: 8, lr: 1e-2, seed: 9, rng: () => random() });
  try {
    for (let i = 0; i < 5; i++) await a.step();
    a.valHistory.push({ step: 0, loss: 2.4 });
    const snapshot = await a.snapshot();
    await b.step(); // 已有 Adam 矩变量，覆盖存档时应该释放旧变量
    const before = tf.memory().numTensors;
    await b.restore(snapshot);
    const restored = await b.snapshot();
    expect(tf.memory().numTensors).toBe(before);
    expect(restored).toEqual(snapshot);
    expect(b.state).toBe('paused');
    expect(b.stats().step).toBe(5);
    expect(b.stats().initialLoss).toBe(snapshot.lossHistory[0]);
    expect(b.stats().ema).toBe(a.stats().ema);

    random = mulberry32(42);
    const expectedLoss = (await a.step())!;
    random = mulberry32(42);
    expect(Math.abs((await b.step())! - expectedLoss)).toBeLessThan(1e-4);
    // step() 返回的是更新前的 loss，再比较更新后的模型以验证 Adam 动量。
    expect(await b.evalValLoss()).toBeCloseTo(await a.evalValLoss(), 5);
    expect(snapshot.lossHistory).toHaveLength(5);
  } finally {
    b.dispose();
  }
});

test('训练中恢复第 0 步快照会先暂停，继续时不重复存档步的里程碑', async () => {
  const milestones: number[] = [];
  const t = make({ onMilestone: ({ step }) => { milestones.push(step); } });
  t.start();
  const saved = await t.runBetweenSteps(() => t.snapshot()); // 排在第 0 步里程碑之后、第一步之前
  expect(saved.step).toBe(0);
  expect(saved.valHistory.map((v) => v.step)).toEqual([0]);
  await untilStep(2);
  await t.restore(saved);
  expect(t.state).toBe('paused');
  expect(await t.snapshot()).toEqual(saved);
  t.start();
  await untilStep(1);
  await t.pause();
  expect(milestones).toEqual([0]);
});

test('载入训练前保存的存档（还没有验证 loss）后继续：第 0 步里程碑做且只做一次', async () => {
  const milestones: number[] = [];
  const t = make({ onMilestone: ({ step }) => { milestones.push(step); } });
  const saved = await t.snapshot();
  expect(saved.valHistory).toEqual([]);
  await t.restore(saved);
  t.start();
  await untilStep(1);
  await t.pause();
  t.start();
  await untilStep(2);
  await t.pause();
  expect(milestones).toEqual([0]);
  expect(t.valHistory.map((v) => v.step)).toEqual([0]);
});

test('载入第 99 步的存档后继续：第 100 步照常做里程碑', async () => {
  const milestones: number[] = [];
  const t = make({ onMilestone: ({ step }) => { milestones.push(step); } });
  for (let i = 0; i < 99; i++) await t.step();
  t.valHistory.push({ step: 0, loss: 2.4 }, { step: 50, loss: 2.2 });
  await t.restore(await t.snapshot());
  t.start();
  await untilStep(101);
  await t.pause();
  expect(milestones).toEqual([100]);
  expect(t.valHistory.map((v) => v.step)).toEqual([0, 50, 100]);
});

test('里程碑进行中点暂停：状态和计时立刻停下，里程碑做完后不再训练下一步', async () => {
  let finishMilestone!: () => void;
  const gate = new Promise<void>((resolve) => { finishMilestone = resolve; });
  const states: string[] = [];
  const t = make({ onMilestone: () => gate, onState: (s) => states.push(s) });
  t.start();
  await vi.waitFor(() => expect(t.valHistory).toHaveLength(1)); // 第 0 步里程碑在等采样
  const paused = t.pause();
  expect(t.state).toBe('paused'); // 不等里程碑做完
  expect(states).toEqual(['running', 'paused']);
  const { elapsedMs } = t.stats();
  await sleep(30);
  expect(t.stats().elapsedMs).toBe(elapsedMs);
  finishMilestone();
  await paused;
  expect(t.stats().step).toBe(0);
  expect(states).toEqual(['running', 'paused']);
});

test('暂停还没生效（里程碑进行中）时再点继续：接着训练，状态回到 running', async () => {
  let finishMilestone!: () => void;
  const gate = new Promise<void>((resolve) => { finishMilestone = resolve; });
  const states: string[] = [];
  const t = make({ onMilestone: () => gate, onState: (s) => states.push(s) });
  t.start();
  await vi.waitFor(() => expect(t.valHistory).toHaveLength(1));
  void t.pause();
  t.start();
  expect(t.state).toBe('running');
  finishMilestone();
  await untilStep(3);
  await t.pause();
  expect(states).toEqual(['running', 'paused', 'running', 'paused']);
  const { step } = t.stats();
  await sleep(50);
  expect(t.stats().step).toBe(step); // 只有一个循环，暂停后不会再有别的循环偷跑
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

test('runBetweenSteps：idle 时立即执行；训练中排到两步之间执行，返回值能拿到，之后继续训练', async () => {
  const t = make();
  expect(await t.runBetweenSteps(async () => t.state)).toBe('idle');
  expect(t.stats().step).toBe(0);

  const step = t.step.bind(t);
  let inStep = false;
  vi.spyOn(t, 'step').mockImplementation(async () => {
    inStep = true;
    try {
      await sleep(10); // 让一步跨过几个宏任务，下面才能在“步进行中”时插队
      return await step();
    } finally {
      inStep = false;
    }
  });
  t.start();
  await untilStep(2);
  await vi.waitFor(() => expect(inStep).toBe(true), { interval: 1 });
  const seen: boolean[] = [];
  const result = await t.runBetweenSteps(async () => {
    seen.push(inStep);
    await sleep(30); // 期间训练循环不能偷跑下一步
    seen.push(inStep);
    return 42;
  });
  expect(result).toBe(42);
  expect(seen).toEqual([false, false]);
  const after = t.stats().step;
  await untilStep(after + 2);
  expect(t.state).toBe('running');
});

test('暂停时跳过排在采样之后、尚未开始的训练步', async () => {
  const t = make();
  let finishSample!: () => void;
  const sampleGate = new Promise<void>((resolve) => { finishSample = resolve; });
  t.start();
  const sample = t.runBetweenSteps(() => sampleGate);
  // 第 0 步的里程碑完成后，第一步会排在本次采样之后。
  await vi.waitFor(() => expect(t.valHistory).toHaveLength(1));
  const paused = t.pause();
  finishSample();
  await sample;
  await paused;
  expect(t.stats().step).toBe(0);
});

test('reset() 先暂停，重新初始化权重和 Adam，清空历史；下一次 start 重新做第 0 步里程碑', async () => {
  const milestones: number[] = [];
  const states: string[] = [];
  const t = make({ onMilestone: ({ step }) => { milestones.push(step); }, onState: (s) => states.push(s) });
  const fresh = tf.memory().numTensors;
  t.start();
  await untilStep(3);
  const trained = Array.from(await t.params.wte.data());
  await t.reset();
  expect(t.state).toBe('idle');
  expect(states.at(-1)).toBe('idle');
  expect(t.stats()).toMatchObject({ step: 0, initialLoss: null, currentLoss: null, ema: null, elapsedMs: 0 });
  expect(t.lossHistory).toEqual([]);
  expect(t.valHistory).toEqual([]);
  expect(tf.memory().numTensors).toBe(fresh);
  expect(Array.from(await t.params.wte.data())).not.toEqual(trained);

  t.start();
  await untilStep(1);
  await t.pause();
  expect(milestones).toEqual([0, 0]);
});

test('loss 出现 NaN 时自动停下，状态为 nan，不记进历史', async () => {
  const states: string[] = [];
  const t = make({ onState: (s) => states.push(s) });
  await t.step();
  await t.step();
  t.params.wte.assign(tf.fill(t.params.wte.shape, NaN));
  const spy = vi.spyOn(t, 'step');
  t.start();
  await vi.waitFor(() => expect(t.state).toBe('nan'));
  const calls = spy.mock.calls.length;
  await sleep(50);
  expect(spy.mock.calls.length).toBe(calls);
  expect(calls).toBe(1);
  expect(t.stats().step).toBe(2);
  expect(states).toEqual(['running', 'nan']);
});

test('microBatch：分块求梯度再累加，与整个 batch 一步的 loss、权重和验证 loss 相同', async () => {
  const whole = make();
  const split = new Trainer({ cfg: MICRO, train: SEQS, val: SEQS.slice(0, 4), bosId: BOS,
    batchSize: 8, lr: 1e-2, seed: 1, rng: mulberry32(1), microBatch: 3 }); // 8 行切成 3、3、2，最后一块不满
  try {
    for (let i = 0; i < 2; i++) expect((await split.step())!).toBeCloseTo((await whole.step())!, 5);
    const tensors = tf.memory().numTensors; // Adam 矩变量在第一步就建好了；之后累加梯度不能留下张量
    expect((await split.step())!).toBeCloseTo((await whole.step())!, 5);
    expect(tf.memory().numTensors).toBe(tensors);
    for (const [name, w] of Object.entries(whole.params)) {
      const a = await w.data();
      const b = await split.params[name].data();
      expect(Math.max(...a.map((x, i) => Math.abs(x - b[i])))).toBeLessThan(1e-5);
    }
    expect(await split.evalValLoss()).toBeCloseTo(await whole.evalValLoss(), 5); // 验证集 4 首切成 3、1
  } finally {
    split.dispose();
  }
});

test('microBatch：一步算到一半点暂停，放弃这一步（权重不变、不留张量），之后照常继续', async () => {
  const base = mulberry32(1);
  let pauseOnNextBatch = false;
  let paused: Promise<void> | undefined;
  let before: { wte: number[]; numTensors: number } | undefined;
  const t: Trainer = make({
    microBatch: 2,
    onStep: (s) => {
      if (s.step !== 2) return;
      pauseOnNextBatch = true;
      before = { wte: Array.from(t.params.wte.dataSync()), numTensors: tf.memory().numTensors };
    },
    // 第 3 步一开始抽样时就点暂停：这时这一步的 4 块一块都还没算
    rng: () => {
      if (pauseOnNextBatch) {
        pauseOnNextBatch = false;
        paused = t.pause();
      }
      return base();
    },
  });
  t.start();
  await vi.waitFor(() => expect(paused).toBeDefined(), { timeout: 10_000 });
  expect(t.state).toBe('paused');
  await paused;
  expect(t.stats().step).toBe(2);
  expect(t.lossHistory).toHaveLength(2);
  expect(Array.from(t.params.wte.dataSync())).toEqual(before!.wte);
  expect(tf.memory().numTensors).toBe(before!.numTensors);

  t.start();
  await untilStep(3);
  await t.pause();
  expect(t.lossHistory.every(Number.isFinite)).toBe(true);
});
