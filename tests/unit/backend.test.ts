import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { expect, test } from 'vitest';
import { backendOrder, chooseBackend, initBackend, smokeTest, type BackendName } from '../../src/backend';

test('backendOrder：auto 或缺省时为 webgpu → webgl → cpu', () => {
  const auto = { order: ['webgpu', 'webgl', 'cpu'], notices: [] };
  expect(backendOrder('')).toEqual(auto);
  expect(backendOrder('?backend=auto')).toEqual(auto);
});

test('backendOrder：指定后端时从它开始往下', () => {
  expect(backendOrder('?backend=webgl')).toEqual({ order: ['webgl', 'cpu'], notices: [] });
  expect(backendOrder('?backend=cpu')).toEqual({ order: ['cpu'], notices: [] });
});

test('backendOrder：未知值按 auto 处理并提示', () => {
  expect(backendOrder('?backend=foo')).toEqual({
    order: ['webgpu', 'webgl', 'cpu'],
    notices: ['未知后端 foo，已自动选择'],
  });
});

/** 只有 ok 里的后端能初始化成功 */
const only = (...ok: BackendName[]) => async (b: BackendName) => ok.includes(b);

test('chooseBackend：webgpu 失败时降级到 webgl 并提示', async () => {
  expect(await chooseBackend(['webgpu', 'webgl', 'cpu'], only('webgl', 'cpu'))).toEqual({
    backend: 'webgl',
    notices: ['WebGPU 不可用，已改用 WebGL'],
  });
});

test('chooseBackend：首选成功时没有提示；抛错也算失败', async () => {
  expect(await chooseBackend(['webgl', 'cpu'], only('webgl'))).toEqual({ backend: 'webgl', notices: [] });
  const throwsOnGpu = async (b: BackendName) => {
    if (b === 'webgpu') throw new Error('no adapter');
    return true;
  };
  expect((await chooseBackend(['webgpu', 'webgl', 'cpu'], throwsOnGpu)).backend).toBe('webgl');
});

test('chooseBackend：只有 cpu 成功时提示非常慢', async () => {
  const r = await chooseBackend(['webgpu', 'webgl', 'cpu'], only('cpu'));
  expect(r.backend).toBe('cpu');
  expect(r.notices).toContain('未使用 GPU 加速，训练会非常慢');
});

test('chooseBackend：全部失败时 reject', async () => {
  await expect(chooseBackend(['webgpu', 'webgl', 'cpu'], only())).rejects.toThrow('没有可用的计算后端');
});

test('smokeTest：cpu 后端下通过，且不留下张量', async () => {
  await tf.setBackend('cpu');
  const before = tf.memory().numTensors;
  expect(smokeTest()).toBe(true);
  expect(tf.memory().numTensors).toBe(before);
});

test('initBackend：?backend=cpu 时用 cpu 并提示非常慢', async () => {
  expect(await initBackend('?backend=cpu')).toEqual({ backend: 'cpu', notices: ['未使用 GPU 加速，训练会非常慢'] });
});
