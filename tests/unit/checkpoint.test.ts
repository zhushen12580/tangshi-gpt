import { expect, test, vi } from 'vitest';
import {
  deleteCheckpoint, isCompatible, loadCheckpoint, makeCheckpoint, saveCheckpoint,
  StorageUnavailableError, storageErrorMessage,
} from '../../src/checkpoint';

const config = { vocabSize: 11, ctx: 8, nLayer: 2, nHead: 2, dModel: 16, dFF: 32 };
const snapshot = {
  step: 0, weights: { wte: new Float32Array([0.1]) }, optimizer: [],
  lossHistory: [], valHistory: [], elapsedMs: 0,
};
const checkpoint = () => makeCheckpoint(snapshot, '01234567', config, [], 123);

test('存档保留二进制权重，只兼容同版本、字表和模型配置', () => {
  const c = checkpoint();
  expect(c).toEqual({ ...snapshot, version: 1, config, vocabHash: '01234567', samples: [], savedAt: 123 });
  expect(isCompatible(c, c.vocabHash, { ...config })).toBe(true);
  expect(isCompatible(c, '76543210', config)).toBe(false);
  for (const key of Object.keys(config) as (keyof typeof config)[]) {
    expect(isCompatible(c, c.vocabHash, { ...config, [key]: config[key] + 1 })).toBe(false);
  }
  const futureVersion = structuredClone(c);
  Object.assign(futureVersion, { version: 2 });
  expect(isCompatible(futureVersion, c.vocabHash, config)).toBe(false);
});

test('存档复制生成记录，之后改动原记录不影响存档', () => {
  const sample = {
    step: 0, valLoss: null, temperature: 0.8, constrained: true, manual: false,
    text: '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。', formatOk: true,
  };
  const samples = [sample];
  const c = makeCheckpoint(snapshot, '01234567', config, samples, 123);
  sample.text = 'x';
  samples.push({ ...sample });
  expect(c.samples).toHaveLength(1);
  expect(c.samples[0]).not.toBe(sample);
  expect(c.samples[0].text).toBe('白日依山尽，黄河入海流。欲穷千里目，更上一层楼。');
});

test('保存失败时区分存储不可用、配额不足和其他错误', () => {
  expect(storageErrorMessage(new DOMException('x', 'QuotaExceededError'))).toBe('无法保存：浏览器存储空间不足');
  expect(storageErrorMessage(new StorageUnavailableError())).toBe('无法保存：此浏览器不支持本地存储');
  expect(storageErrorMessage(new Error('x'))).toBe('无法保存：x');
});

test('IndexedDB 不可用时拒绝保存，读取和删除按无存档处理', async () => {
  vi.stubGlobal('indexedDB', undefined);
  try {
    await expect(saveCheckpoint(checkpoint(), undefined)).rejects.toBeInstanceOf(StorageUnavailableError);
    await expect(loadCheckpoint(undefined)).resolves.toBeNull();
    await expect(deleteCheckpoint(undefined)).resolves.toBeUndefined();
  } finally {
    vi.unstubAllGlobals();
  }
});
