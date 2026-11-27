import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { readFileSync } from 'node:fs';
import { expect, test } from 'vitest';
import { traceErrors, type Trace } from '../../explainer/timeline';
import { createDataset, splitTrainVal } from '../../src/data';
import { overfitStatus } from '../../src/explain';
import { MODEL_DEFAULTS, countParams, initParams } from '../../src/model';
import { BATCH_SIZE, LEARNING_RATE } from '../../src/train';

// 入库的录制结果必须和现在的代码、数据对得上：改了数据或模型就要重新录（npm run build:explainer-trace）
const trace: Trace = JSON.parse(readFileSync('explainer/trace.json', 'utf8'));
const ds = createDataset(JSON.parse(readFileSync('public/data/poems.json', 'utf8')));

test('meta 里的数字等于用代码和 poems.json 现算的值', () => {
  const V = ds.vocab.length;
  const p = initParams({ ...MODEL_DEFAULTS, vocabSize: V }, 1);
  const params = countParams(p);
  tf.dispose(Object.values(p));
  const { meta } = trace;
  expect([meta.vocab, meta.poems, meta.valCount, meta.params]).toEqual([V, ds.poems.length, ds.val.length, params]);
  expect([meta.batch, meta.lr]).toEqual([BATCH_SIZE, LEARNING_RATE]);
  expect(meta.lnVocab).toBe(Number(Math.log(V).toFixed(2)));
  expect(['webgpu', 'webgl']).toContain(meta.backend);
});

test('训练 1500 步，每 50 步一个验证点，best 是验证 loss 的最低点，之后确实回升', () => {
  expect(trace.meta.steps).toBe(1500);
  expect(trace.loss).toHaveLength(1500);
  expect(trace.val.map((v) => v.step)).toEqual(Array.from({ length: 31 }, (_, i) => i * 50));
  expect(trace.val).toContainEqual(trace.best);
  expect(trace.best.loss).toBe(Math.min(...trace.val.map((v) => v.loss)));
  expect(overfitStatus(trace.val)).not.toBeNull();
});

test('每个里程碑一首样本，inTrain 等于「训练集原文里有这首诗」', () => {
  const trainSet = new Set(splitTrainVal(ds.poems).train);
  expect(trace.samples.map((s) => s.step)).toEqual(trace.val.map((v) => v.step));
  for (const s of trace.samples) expect(s.inTrain).toBe(trainSet.has(s.text));
});

test('注意力是 3×4×24×24，每行的权重加起来约等于 1', () => {
  const { nLayer, nHead, ctx } = MODEL_DEFAULTS;
  expect(trace.attention).toHaveLength(nLayer);
  for (const layer of trace.attention) {
    expect(layer).toHaveLength(nHead);
    for (const head of layer) {
      expect(head).toHaveLength(ctx);
      for (const row of head) {
        expect(row).toHaveLength(ctx);
        expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(0.02);
      }
    }
  }
});

test('traceErrors：入库的 trace 没有问题；没有回升或缺字段时报错', () => {
  expect(traceErrors(trace)).toEqual([]);
  const falling = { ...trace, val: trace.val.map((v, i) => ({ step: v.step, loss: 9 - i * 0.1 })) };
  expect(traceErrors(falling)).toContain('验证 loss 没有回升');
  const { attention: _, ...noAttention } = trace;
  expect(traceErrors(noAttention as Trace)).not.toEqual([]);
});
