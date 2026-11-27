// 原理视频的数据：一次真实训练的录制结果（explainer/trace.json）的类型和检查。录制页和单测共用

import { overfitStatus } from '../src/explain';
import { MODEL_DEFAULTS } from '../src/model';
import { isWujue } from '../src/sample';
import { MILESTONE_EVERY } from '../src/train';

export interface Point { step: number; loss: number }
/** 候选字和模型给它的概率 */
export interface Candidate { char: string; p: number }

export interface Trace {
  meta: {
    backend: 'webgpu' | 'webgl';
    steps: number;
    batch: number;
    lr: number;
    vocab: number; // 字表大小 V（含 <s>）
    lnVocab: number; // ln V，两位小数
    params: number;
    poems: number;
    valCount: number;
    recordedAt: string;
  };
  loss: number[]; // 每步的训练 loss，3 位小数
  val: Point[]; // 每个里程碑的验证 loss
  best: Point; // 验证 loss 最低的里程碑，下面的 demo、guess、temperature、attention 都用这一步的模型
  samples: { step: number; val: number; text: string; inTrain: boolean }[]; // 每个里程碑一首（温度 0.8，带格式约束）
  demo: { text: string; top: Candidate[][] }; // 现写的一首，每个位置抽字时的前 3 个候选
  guess: {
    context: string; // 跟在 <s> 后面的上文
    answer: string;
    step0: { p: number; top: Candidate[] }; // 第 0 步的模型：答案的概率和前 5 个候选（温度 1，不加掩码）
    best: { p: number; top: Candidate[] };
  };
  temperature: { temperature: number; top: Candidate[]; rest: number }[]; // 第 1 个字：前 8 个候选和「其余」合计
  attention: number[][][][]; // 《登鹳雀楼》的注意力 [层][头][查询位置][键位置]，3 位小数
}

const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const isPoint = (p: unknown): p is Point => isNum((p as Point)?.step) && isNum((p as Point)?.loss);
const isCands = (c: unknown, min: number, max: number) =>
  Array.isArray(c) && c.length >= min && c.length <= max && c.every((x) => typeof x?.char === 'string' && isNum(x.p));
const isMatrix = (a: unknown, n: number, inner: (x: unknown) => boolean) => Array.isArray(a) && a.length === n && a.every(inner);

/** 录到的东西能不能拿来做视频：没有问题时返回空数组。trace 来自页面或 JSON 文件，字段可能缺 */
export function traceErrors(t: Trace): string[] {
  const { nLayer, nHead, ctx } = MODEL_DEFAULTS;
  const m = t?.meta;
  const steps = m?.steps;
  const val = t?.val;
  const checks: [unknown, string][] = [
    [m?.backend === 'webgpu' || m?.backend === 'webgl', 'meta.backend 必须是 webgpu 或 webgl'],
    [m && (['steps', 'batch', 'lr', 'vocab', 'lnVocab', 'params', 'poems', 'valCount'] as const).every((k) => isNum(m[k])), 'meta 缺少数值'],
    [Array.isArray(t?.loss) && t.loss.length === steps && t.loss.every(isNum), 'loss 要有 meta.steps 个有限值'],
    [Array.isArray(val) && val.length === steps / MILESTONE_EVERY + 1 && val.every((p, i) => isPoint(p) && p.step === i * MILESTONE_EVERY),
      `val 要在第 0、${MILESTONE_EVERY} … ${steps} 步各有一个点`],
    [isPoint(t?.best) && Array.isArray(val) && val.every(isPoint) && val.some((p) => p.step === t.best.step && p.loss === t.best.loss)
      && val.every((p) => p.loss >= t.best.loss), 'best 要是 val 里的最低点'],
    [Array.isArray(t?.samples) && t.samples.length === val?.length && t.samples.every((s, i) =>
      s?.step === val[i]?.step && s?.val === val[i]?.loss && isWujue(s?.text) && typeof s?.inTrain === 'boolean'), 'samples 要和 val 一一对应'],
    [isWujue(t?.demo?.text) && isMatrix(t.demo.top, ctx, (c) => isCands(c, 1, 3)), 'demo 要有一首五绝和每个位置的候选'],
    [typeof t?.guess?.context === 'string' && typeof t.guess.answer === 'string'
      && [t.guess.step0, t.guess.best].every((g) => isNum(g?.p) && isCands(g.top, 5, 5)), 'guess 缺字段'],
    [isMatrix(t?.temperature, 2, (x) => {
      const e = x as Trace['temperature'][number];
      return isNum(e?.temperature) && isNum(e.rest) && isCands(e.top, 8, 8);
    }), 'temperature 要有两组前 8 个候选'],
    [isMatrix(t?.attention, nLayer, (l) => isMatrix(l, nHead, (h) => isMatrix(h, ctx, (r) => isMatrix(r, ctx, isNum)))),
      `attention 要是 ${nLayer}×${nHead}×${ctx}×${ctx}`],
  ];
  const errors = checks.filter(([ok]) => !ok).map(([, msg]) => msg);
  if (Array.isArray(val) && val.every(isPoint) && overfitStatus(val) === null) errors.push('验证 loss 没有回升');
  return errors;
}
