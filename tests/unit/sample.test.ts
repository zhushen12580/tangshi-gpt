import { expect, test, vi } from 'vitest';
import {
  formatMeta,
  isWujue,
  maskLogits,
  MAX_SAMPLES,
  pushSample,
  sampleFromLogits,
  samplePoem,
  type SampleRecord,
  type TokenIds,
} from '../../src/sample';
import { mulberry32 } from './rng';

// 伪造的字表：id 0–6 是普通字，7 = <s>，8 = 「，」，9 = 「。」
const IDS: TokenIds = { bos: 7, comma: 8, period: 9 };
const V = 10;

test('isWujue：标准五绝为真，标点对调或少一个字为假', () => {
  expect(isWujue('白日依山尽，黄河入海流。欲穷千里目，更上一层楼。')).toBe(true);
  expect(isWujue('白日依山尽。黄河入海流，欲穷千里目。更上一层楼，')).toBe(false);
  expect(isWujue('白日依山尽，黄河入海流。欲穷千里目，更上一层。')).toBe(false);
});

test('maskLogits：不改原数组；<s> 永远屏蔽；约束时标点位只留对应标点、其他位置屏蔽标点', () => {
  const logits = new Float32Array(V).fill(1);
  const at = (pos: number, constrain: boolean) => maskLogits(logits, pos, IDS, constrain);

  const p5 = at(5, true);
  expect([...logits]).toEqual(new Array(V).fill(1));
  expect(p5[IDS.bos]).toBe(-Infinity);
  expect([...p5].map((x, i) => (Number.isFinite(x) ? i : -1)).filter((i) => i >= 0)).toEqual([IDS.comma]);
  expect([...at(11, true)].map((x, i) => (Number.isFinite(x) ? i : -1)).filter((i) => i >= 0)).toEqual([IDS.period]);

  const p0 = at(0, true);
  expect(p0[IDS.comma]).toBe(-Infinity);
  expect(p0[IDS.period]).toBe(-Infinity);
  expect(p0[0]).toBe(1);

  const free = at(5, false);
  expect([...free].filter((x) => x === -Infinity)).toHaveLength(1);
  expect(free[IDS.bos]).toBe(-Infinity);
});

test('sampleFromLogits：温度 0.1 配极大 logits 不出 NaN；-Infinity 不会被抽到；下标总在范围内', () => {
  expect(sampleFromLogits(new Float32Array([1000, 0, -1000]), 0.1, () => 0.5)).toBe(0);
  expect(sampleFromLogits(new Float32Array([0, 0, -Infinity]), 1, () => 0.99)).toBe(1);
  const rng = mulberry32(3);
  for (let k = 0; k < 200; k++) {
    const logits = new Float32Array([5, -Infinity, 1e4, 0]);
    const i = sampleFromLogits(logits, 0.1 + (k % 15) / 10, rng);
    expect(i).toBeGreaterThanOrEqual(0);
    expect(i).toBeLessThan(4);
    expect(i).not.toBe(1);
  }
  expect(sampleFromLogits(new Float32Array([0, 0]), 1, () => 0.999999999)).toBe(1);
});

test('samplePoem：约束时 5/11/17/23 位依次是「，。，。」，其他位置没有标点', async () => {
  const seen: number[][] = [];
  const uniform = async (ids: number[]) => {
    seen.push(ids);
    return new Float32Array(V);
  };
  const ids = await samplePoem(uniform, IDS, { temperature: 0.8, constrain: true, rng: mulberry32(7) });
  expect(ids).toHaveLength(24);
  expect(seen[0]).toEqual([IDS.bos]);
  expect(seen[23]).toHaveLength(24);
  ids.forEach((id, pos) => {
    if (pos % 6 === 5) expect(id).toBe(pos % 12 === 5 ? IDS.comma : IDS.period);
    else expect([IDS.comma, IDS.period, IDS.bos]).not.toContain(id);
  });
});

test('samplePoem：每写一个字让出一次主线程（CPU 后端一首诗要算好几秒，期间点击也要能响应）', async () => {
  let calls = 0;
  const counting = async () => {
    calls++;
    return new Float32Array(V);
  };
  let callsWhenTimerFired = -1;
  setTimeout(() => (callsWhenTimerFired = calls), 0);
  const timers = vi.spyOn(globalThis, 'setTimeout');
  try {
    await samplePoem(counting, IDS, { temperature: 1, constrain: false, rng: mulberry32(1) });
    expect(timers).toHaveBeenCalledTimes(24); // 每个字之后一次
  } finally {
    timers.mockRestore();
  }
  expect(calls).toBe(24);
  expect(callsWhenTimerFired).toBeGreaterThanOrEqual(0); // 真的让出去了：samplePoem 还没写完，别的宏任务就已经执行
  expect(callsWhenTimerFired).toBeLessThan(24);
});

test('samplePoem：不约束、logits 偏向「。」时，第 0 位就是「。」', async () => {
  const biased = async () => {
    const l = new Float32Array(V);
    l[IDS.period] = 50;
    l[IDS.bos] = 100; // <s> 再大也不会被抽到
    return l;
  };
  const ids = await samplePoem(biased, IDS, { temperature: 1, constrain: false, rng: mulberry32(1) });
  expect(ids[0]).toBe(IDS.period);
});

const rec = (step: number, manual = false): SampleRecord => ({
  step,
  valLoss: 5.4321,
  temperature: 0.8,
  constrained: true,
  manual,
  text: '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。',
  formatOk: true,
});

test('pushSample：新的在最前面；超过 200 条先丢最旧的，但第 0 步的自动记录永远保留；不改入参', () => {
  let list = pushSample([], rec(0));
  for (let i = 1; i <= 250; i++) {
    const before = list;
    const len = before.length;
    list = pushSample(before, rec(i * 50, i % 3 === 0));
    expect(before).toHaveLength(len);
  }
  expect(list).toHaveLength(MAX_SAMPLES);
  expect(list[0].step).toBe(250 * 50);
  expect(list.at(-1)).toEqual(rec(0));
  expect(list.at(-2)!.step).toBe(52 * 50); // 第 1–51 条被丢掉
});

test('formatMeta：步数、验证 loss、自动/手动、温度、约束/自由', () => {
  expect(formatMeta(rec(50))).toBe('第 50 步 · 验证 loss 5.432 · 自动 · 温度 0.8 · 约束');
  expect(formatMeta({ ...rec(0, true), valLoss: null })).toBe('第 0 步 · 验证 loss — · 手动 · 温度 0.8 · 约束');
  expect(formatMeta({ ...rec(100), temperature: 1.5, constrained: false })).toBe(
    '第 100 步 · 验证 loss 5.432 · 自动 · 温度 1.5 · 自由',
  );
});
