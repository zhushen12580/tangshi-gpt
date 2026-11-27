// 录制一次真实训练给原理视频用：src/ 里的真实模块和默认常量，从随机初始化训练 1500 步。
// 只在开发服务器上由 scripts/build-explainer.mjs 打开，结果放到 window.__result

import { initBackend } from '../src/backend';
import { decode, encode, loadDataset, splitTrainVal } from '../src/data';
import { MODEL_DEFAULTS, attentionMaps, countParams, nextTokenLogits } from '../src/model';
import presets from '../src/presets.json';
import { maskLogits, samplePoem } from '../src/sample';
import { BATCH_SIZE, LEARNING_RATE, Trainer, type TrainerSnapshot } from '../src/train';
import { traceErrors, type Candidate, type Point, type Trace } from './timeline';

const STEPS = 1500;
const SAMPLE_TEMPERATURE = 0.8; // 和页面滑块的默认值一样
const GUESS_CONTEXT = '白日依山尽，黄';
const GUESS_ANSWER = '河';
const TEMPERATURES = [0.3, 1.5];

declare global {
  interface Window { __result?: { trace: Trace } | { error: string } }
}

const round3 = (x: number) => Math.round(x * 1000) / 1000;
const sig4 = (x: number) => Number(x.toPrecision(4)); // 概率可能很小（第 0 步约 0.03%），保留有效数字而不是小数位
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** softmax(logits / 温度)，先减最大值；被屏蔽（-Infinity）的字概率为 0 */
function softmax(logits: Float32Array, temperature: number): number[] {
  const scaled = Array.from(logits, (x) => x / temperature);
  const max = scaled.reduce((a, b) => Math.max(a, b), -Infinity);
  const e = scaled.map((x) => Math.exp(x - max));
  const total = sum(e);
  return e.map((x) => x / total);
}

/** 概率最高的 k 个字（概率为 0 的不算），概率不取整 */
const top = (probs: number[], vocab: string[], k: number): Candidate[] =>
  probs.map((p, i) => ({ char: vocab[i], p })).filter((c) => c.p > 0).sort((a, b) => b.p - a.p).slice(0, k);
const rounded = (cs: Candidate[]) => cs.map((c) => ({ char: c.char, p: sig4(c.p) }));

async function record(): Promise<Trace> {
  const { backend } = await initBackend(location.search);
  if (backend === 'cpu') throw new Error('录制需要 WebGPU 或 WebGL 加速');
  const ds = await loadDataset('/data/poems.json');
  const cfg = { ...MODEL_DEFAULTS, vocabSize: ds.vocab.length };
  const ids = { bos: ds.bosId, comma: ds.stoi.get('，')!, period: ds.stoi.get('。')! };
  const trainSet = new Set(splitTrainVal(ds.poems).train);
  const logits = (seq: number[]) => nextTokenLogits(trainer.params, cfg, seq);

  // 「白日依山尽，黄」后面猜「河」：温度 1，不加掩码
  const guess = async () => {
    const probs = softmax(await logits([ds.bosId, ...encode(GUESS_CONTEXT, ds.stoi)]), 1);
    return { p: sig4(probs[ds.stoi.get(GUESS_ANSWER)!]), top: rounded(top(probs, ds.vocab, 5)) };
  };

  const val: Point[] = [];
  const samples: Trace['samples'] = [];
  let best: { point: Point; snapshot: TrainerSnapshot } | null = null;
  let finish!: () => void;
  let fail!: (err: Error) => void;
  const finished = new Promise<void>((resolve, reject) => ((finish = resolve), (fail = reject)));

  const trainer = new Trainer({
    cfg, train: ds.train, val: ds.val, bosId: ds.bosId,
    onState: (s) => s === 'nan' && fail(new Error(`第 ${trainer.lossHistory.length + 1} 步出现 NaN`)),
    onMilestone: async ({ step, valLoss }) => {
      const text = decode(await samplePoem(logits, ids, { temperature: SAMPLE_TEMPERATURE, constrain: true }), ds.vocab);
      const point = { step, loss: round3(valLoss) };
      val.push(point);
      samples.push({ step, val: point.loss, text, inTrain: trainSet.has(text) });
      // onMilestone 本来就在两步之间执行，直接拍快照；这里再用 runBetweenSteps 会等自己，死锁
      if (!best || point.loss < best.point.loss) best = { point, snapshot: await trainer.snapshot() };
      if (step % 100 === 0) console.log(`第 ${step}/${STEPS} 步 · 训练 loss ${trainer.stats().currentLoss?.toFixed(3) ?? '—'} · 验证 loss ${point.loss.toFixed(3)}`);
      if (step === STEPS) void trainer.pause().then(finish); // 只发出暂停请求，循环在这一步之后退出
    },
  });

  const step0 = await guess();
  trainer.start();
  await finished;
  const loss = trainer.lossHistory.map(round3);
  const { point, snapshot } = best!;
  console.log(`验证 loss 最低 ${point.loss.toFixed(3)}（第 ${point.step} 步），最后 ${val.at(-1)!.loss.toFixed(3)}`);
  await trainer.restore(snapshot);

  // 最好的模型现写一首，记下每个位置抽字时（带掩码、同一温度）的前 3 个候选
  const seen: Float32Array[] = [];
  const demoIds = await samplePoem(async (seq) => {
    const l = await logits(seq);
    seen.push(l);
    return l;
  }, ids, { temperature: SAMPLE_TEMPERATURE, constrain: true });
  const demo = {
    text: decode(demoIds, ds.vocab),
    top: seen.map((l, pos) => rounded(top(softmax(maskLogits(l, pos, ids, true), SAMPLE_TEMPERATURE), ds.vocab, 3))),
  };

  const first = maskLogits(await logits([ds.bosId]), 0, ids, true);
  const temperature = TEMPERATURES.map((t) => {
    const cands = top(softmax(first, t), ds.vocab, 8);
    return { temperature: t, top: rounded(cands), rest: sig4(Math.max(0, 1 - sum(cands.map((c) => c.p)))) };
  });

  const poem = [...presets.find((p) => p.title === '登鹳雀楼')!.poem].slice(0, cfg.ctx - 1).join('');
  const maps = await attentionMaps(trainer.params, cfg, [ds.bosId, ...encode(poem, ds.stoi)]);

  return {
    meta: {
      backend, steps: STEPS, batch: BATCH_SIZE, lr: LEARNING_RATE,
      vocab: ds.vocab.length, lnVocab: Number(Math.log(ds.vocab.length).toFixed(2)),
      params: countParams(trainer.params), poems: ds.poems.length, valCount: ds.val.length,
      recordedAt: new Date().toISOString(),
    },
    loss, val, best: point, samples, demo,
    guess: { context: GUESS_CONTEXT, answer: GUESS_ANSWER, step0, best: await guess() },
    temperature,
    attention: maps.map((l) => l.map((h) => h.map((r) => r.map(round3)))),
  };
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
// 训练循环里的错误不会传到 record()，从这里接住，免得脚本一直等
window.addEventListener('unhandledrejection', (e) => (window.__result ??= { error: message(e.reason) }));
record().then(
  (trace) => {
    const errors = traceErrors(trace);
    window.__result = errors.length ? { error: errors.join('；') } : { trace };
  },
  (err) => (window.__result = { error: message(err) }),
);
