// 原理视频的数据：一次真实训练的录制结果（explainer/trace.json）的类型和检查。录制页和单测共用

import { guessCount, overfitStatus } from '../src/explain';
import { MODEL_DEFAULTS } from '../src/model';
import { isWujue } from '../src/sample';
import presets from '../src/presets.json';
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

// ---- 分镜和字幕：规格「原理视频 · 分镜与字幕」，{} 里的数一律从 trace 或代码取 ----

export const TITLE = '唐诗 GPT：1 分钟看懂它怎么学会写诗';
export const SITE = 'zhushen12580.github.io/tangshi-gpt';
export const BACKEND_LABEL = { webgpu: 'WebGPU', webgl: 'WebGL' } as const;
const MOTIFS = ['月', '秋风', '江', '花', '酒'];
const CAPTION_MIN = 2; // 每条字幕至少显示的秒数
// 每条显示 max(CAPTION_MIN, 字数 / 字速) 秒。先按 7.5 字/秒排（计划的 7 总长 91.7 秒超了 90）；
// 重录后字幕变长、超过 90 秒时，再按规格下限 8 字/秒排（review 发现 F-b4ea32f7f59f）
const CHARS_PER_SECOND = [7.5, 8];
const DURATION = { min: 60, max: 90 }; // 总长（秒）
export const FPS = 30; // 视频帧率，render.ts 按 i / FPS 秒画第 i 帧
export const TRANSITION = 0.5; // 分镜之间的过渡秒数
export const PEAK_ROW = 10; // 注意力分镜看第 10 位「海」那一行

export interface Caption { text: string; start: number; end: number }
export interface Scene { id: number; name: string; start: number; end: number; captions: Caption[] }
export interface Timeline { scenes: Scene[]; duration: number }

/** 概率 ≥ 10% 写整数百分数，否则两位有效数字 */
export const formatPercent = (p: number): string => (p >= 0.1 ? `${Math.round(p * 100)}%` : `${(p * 100).toPrecision(2)}%`);
const f2 = (x: number) => x.toFixed(2);
const nll = (p: number) => f2(-Math.log(p));

/** 《登鹳雀楼》的注意力输入：<s> 加上诗的前 ctx − 1 个字 */
export const attentionTokens = (): string[] =>
  ['<s>', ...presets.find((p) => p.title === '登鹳雀楼')!.poem].slice(0, MODEL_DEFAULTS.ctx);

/** 「海」那一行在全部层、头和第 1–9 列（不含 <s> 和「海」自己）里最大的权重；layer、head 从 0 数 */
export function attentionPeak(t: Trace): { layer: number; head: number; col: number; w: number } {
  let peak = { layer: 0, head: 0, col: 1, w: -1 };
  t.attention.forEach((layer, l) => layer.forEach((head, h) => {
    for (let col = 1; col < PEAK_ROW; col++) if (head[PEAK_ROW][col] > peak.w) peak = { layer: l, head: h, col, w: head[PEAK_ROW][col] };
  }));
  return peak;
}

/**
 * 「对答案」的最后一句：最好的模型是按整套验证集的 loss 选的，「河」这一个字的概率不一定升，
 * 措辞跟着实测方向走；显示出来的数相同就说「还是」（review 发现 F-f9e0582eeb7d）
 */
function guessResult(t: Trace): string {
  const { step0, best } = t.guess;
  const up = best.p > step0.p;
  const [p0, p1, l0, l1] = [formatPercent(step0.p), formatPercent(best.p), nll(step0.p), nll(best.p)];
  const prob = p0 === p1 ? `概率还是 ${p1}` : up ? `概率升到 ${p1}` : `概率反而降到 ${p1}`;
  const cost = l0 === l1 ? `扣分还是 ${l1}` : up ? `只扣 ${l1}` : `要扣 ${l1}`;
  const why = up || p0 === p1 ? '' : '：整体猜得更准，不等于每个字都更准';
  return `练到第 ${t.best.step} 步，${prob}，${cost}${why}。`;
}

/** 每个分镜的字幕文字，按规格的分镜表填数 */
function sceneTexts(t: Trace): [string, string[]][] {
  const { meta, best, guess } = t;
  const motifs = MOTIFS.filter((m) => t.samples.some((s) => s.step <= best.step && s.text.includes(m)));
  const copy = t.samples.find((s) => s.step > best.step && s.inTrain);
  const peak = attentionPeak(t);
  const [low, high] = t.temperature;
  return [
    ['片头', [TITLE, `下面的数字都来自一次真实训练（${BACKEND_LABEL[meta.backend]}，${meta.steps} 步）`]],
    ['猜下一个字', ['它只做一件事：根据前面的字，猜下一个字。', '从 <s>（开头记号）开始，一个字一个字猜满 24 个，就是一首五绝。']],
    ['对答案、扣分', [
      `训练时拿真唐诗对答案：这里正确的字是「${guess.answer}」。`,
      `刚开始，它给「${guess.answer}」的概率只有 ${formatPercent(guess.step0.p)}，扣分 loss = −ln(概率) ≈ ${nll(guess.step0.p)}。`,
      guessResult(t),
    ]],
    ['微调参数', [
      `每一步，把全部约 ${Math.round(meta.params / 1e4)} 万个参数（${meta.params}）都往“猜对”的方向挪一点。`,
      `每步一起猜 ${meta.batch} 首诗，这次一共重复了 ${meta.steps} 步。`,
    ]],
    ['loss 下降', [
      `一开始 loss ≈ ln ${meta.vocab} ≈ ${f2(meta.lnVocab)}：相当于在 ${meta.vocab} 个字里瞎蒙。`,
      `到第 ${best.step} 步，验证 loss 降到 ${f2(best.loss)}，≈ 在 ${guessCount(best.loss)} 个字里蒙；${motifs.length ? motifs.join('、') : '常用字'}开始出现。`,
    ]],
    ['训练 loss 与验证 loss', [
      `验证 loss 是模拟考：${meta.valCount} 首诗藏起来，从不给它学。`,
      `训练 loss 一直降，最后是 ${f2(t.loss.at(-1)!)}；验证 loss 在第 ${best.step} 步最低（${f2(best.loss)}），之后回升到 ${f2(t.val.at(-1)!.loss)}。`,
      `它开始背诗了（过拟合）。${copy ? `第 ${copy.step} 步写出的那首，就是训练集里的原文。` : ''}`,
    ]],
    ['温度', [
      '同一个概率分布，温度决定它敢不敢挑冷门字。',
      `温度 ${low.temperature}：几乎只挑最稳的几个字；温度 ${high.temperature}：冷门字也常被挑中。`,
    ]],
    ['注意力', [
      '猜下一个字时，它会回头看前面的字。',
      `《登鹳雀楼》里，第 ${peak.layer + 1} 层第 ${peak.head + 1} 个头在「${attentionTokens()[PEAK_ROW]}」这个位置最看重「${attentionTokens()[peak.col]}」（${f2(peak.w)}）。`,
    ]],
    ['片尾', [`打开页面，亲手训练一次：${SITE}`]],
  ];
}

/**
 * 字幕依次出现，每条 max(CAPTION_MIN, 字数 / rate) 秒；分镜之间过渡 TRANSITION 秒。
 * 都按整帧排、向上取整，时刻写成「帧号 / FPS」，和渲染时第 i 帧的 i / FPS 完全相同：
 * 按每秒 8 字排时一个字 3.75 帧，按实数秒排会让字幕少 1 帧（verify 发现 F-a74aa2f447ad）
 */
function layout(texts: [string, string[]][], rate: number): Timeline {
  const toFrames = (seconds: number) => Math.ceil(seconds * FPS - 1e-9); // 1e-9：整帧的秒数乘回来可能多出一点浮点误差
  let at = -toFrames(TRANSITION);
  const scenes = texts.map(([name, captionTexts], id) => {
    const start = (at += toFrames(TRANSITION)) / FPS;
    const captions = captionTexts.map((text) =>
      ({ text, start: at / FPS, end: (at += toFrames(Math.max(CAPTION_MIN, [...text].length / rate))) / FPS }));
    return { id, name, start, end: at / FPS, captions };
  });
  return { scenes, duration: at / FPS };
}

/** 按 CHARS_PER_SECOND 依次试，用第一个不超过 90 秒的；怎么排都不在 60–90 秒内就报错，渲染命令以非 0 退出，不写视频 */
export function buildTimeline(t: Trace): Timeline {
  const texts = sceneTexts(t);
  const tl = CHARS_PER_SECOND.map((rate) => layout(texts, rate)).find((x) => x.duration <= DURATION.max);
  if (!tl) throw new Error(`字幕按每秒 ${CHARS_PER_SECOND.at(-1)} 字排也超过 ${DURATION.max} 秒：只能删减不含数字的字幕，不能改数字`);
  if (tl.duration < DURATION.min) throw new Error(`字幕总长 ${tl.duration.toFixed(1)} 秒，不到 ${DURATION.min} 秒`);
  return tl;
}

/** 文字版：片名，再按分镜写名称和字幕（不含 BOM，写文件时再加） */
export const transcript = (t: Trace): string =>
  [TITLE, ...buildTimeline(t).scenes.map((s) => `【${s.name}】\n${s.captions.map((c) => c.text).join('\n')}`)].join('\n\n') + '\n';
