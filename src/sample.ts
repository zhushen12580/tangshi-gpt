// 采样：温度、五绝格式约束、格式校验，以及生成记录列表

export interface TokenIds {
  bos: number;
  comma: number;
  period: number;
}

export interface SampleRecord {
  step: number;
  valLoss: number | null;
  temperature: number;
  constrained: boolean;
  manual: boolean;
  text: string;
  formatOk: boolean;
}

export const MAX_SAMPLES = 200;
const POEM_LEN = 24;
const WUJUE_RE = /^([一-鿿]{5}，[一-鿿]{5}。){2}$/u;

export function isWujue(text: string): boolean {
  return WUJUE_RE.test(text);
}

/** 返回屏蔽后的新 logits：<s> 永远屏蔽；约束时第 5、17 位只能是「，」，第 11、23 位只能是「。」，其他位置不能是标点 */
export function maskLogits(logits: Float32Array, pos: number, ids: TokenIds, constrain: boolean): Float32Array {
  const out = Float32Array.from(logits);
  out[ids.bos] = -Infinity;
  if (!constrain) return out;
  if (pos % 6 === 5) {
    out.fill(-Infinity);
    out[pos % 12 === 5 ? ids.comma : ids.period] = 0; // 只剩一个有限值，抽中它的概率就是 1
  } else {
    out[ids.comma] = -Infinity;
    out[ids.period] = -Infinity;
  }
  return out;
}

/** logits 除以温度，减去最大值再 softmax（温度很低时也不会溢出成 NaN），按累积概率抽一个下标 */
export function sampleFromLogits(logits: Float32Array, temperature: number, rng: () => number = Math.random): number {
  const scaled = Array.from(logits, (x) => x / temperature);
  const max = scaled.reduce((a, b) => Math.max(a, b), -Infinity);
  const probs = scaled.map((x) => Math.exp(x - max)); // -Infinity 变成 0
  let r = rng() * probs.reduce((a, b) => a + b, 0);
  let last = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] === 0) continue;
    last = i;
    r -= probs[i];
    if (r < 0) return i;
  }
  return last; // 浮点误差让 r 没减到负数时，取最后一个概率不为 0 的下标
}

/** 从 [<s>] 开始自回归生成 24 个 id（不含 <s>） */
export async function samplePoem(
  nextLogits: (ids: number[]) => Promise<Float32Array>,
  ids: TokenIds,
  opts: { temperature: number; constrain: boolean; rng?: () => number },
): Promise<number[]> {
  let seq = [ids.bos];
  for (let pos = 0; pos < POEM_LEN; pos++) {
    const logits = maskLogits(await nextLogits(seq), pos, ids, opts.constrain);
    seq = [...seq, sampleFromLogits(logits, opts.temperature, opts.rng)];
    await new Promise((r) => setTimeout(r, 0)); // 每个字之后让出主线程：CPU 后端一首诗要几秒，期间点击也要能响应
  }
  return seq.slice(1);
}

const isFirstAuto = (r: SampleRecord) => r.step === 0 && !r.manual;

/** 新记录放最前面，最多 MAX_SAMPLES 条；超出时丢掉最旧的一条，但第 0 步的自动记录（“从乱码到唐诗”的起点）永远保留 */
export function pushSample(list: SampleRecord[], rec: SampleRecord): SampleRecord[] {
  const next = [rec, ...list];
  if (next.length <= MAX_SAMPLES) return next;
  let drop = next.length - 1;
  while (drop > 0 && isFirstAuto(next[drop])) drop--;
  return next.filter((_, i) => i !== drop);
}

export function formatMeta(r: SampleRecord): string {
  const val = r.valLoss === null ? '—' : r.valLoss.toFixed(3);
  return `第 ${r.step} 步 · 验证 loss ${val} · ${r.manual ? '手动' : '自动'} · 温度 ${r.temperature.toFixed(1)} · ${r.constrained ? '约束' : '自由'}`;
}
