// 数据：字表、编码、训练/验证切分、取 batch、加载 poems.json

import presets from './presets.json';
import { isWujue } from './sample';

export const BOS = '<s>';

export interface PoemsFile {
  source: string;
  commit: string;
  license: string;
  count: number;
  poems: string[];
}

export interface Dataset {
  poems: string[];
  vocab: string[];
  stoi: Map<string, number>;
  bosId: number;
  train: number[][];
  val: number[][];
  hash: string;
}

/** 字表 = 数据集里所有字符按码点升序，最后加 <s>（id = V−1） */
export function buildVocab(poems: string[]): string[] {
  const chars = [...new Set(poems.join(''))].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
  return [...chars, BOS];
}

/** 按码点切分成 id；遇到字表外的字符直接报错 */
export function encode(text: string, stoi: Map<string, number>): number[] {
  return [...text].map((c) => {
    const id = stoi.get(c);
    if (id === undefined) throw new Error(`字符不在字表中：${c}`);
    return id;
  });
}

export function decode(ids: number[], vocab: string[]): string {
  return ids.map((i) => vocab[i]).join('');
}

/** 确定性切分：下标 % 20 == 0 的进验证集 */
export function splitTrainVal<T>(items: T[]): { train: T[]; val: T[] } {
  return {
    train: items.filter((_, i) => i % 20 !== 0),
    val: items.filter((_, i) => i % 20 === 0),
  };
}

/** 有放回地随机取 batchSize 首：输入 = <s> + 诗的前 T−1 个字，目标 = 整首诗，按行展平 */
export function makeBatch(
  seqs: number[][],
  bosId: number,
  batchSize: number,
  rng: () => number = Math.random,
): { inputs: Int32Array; targets: Int32Array } {
  const T = seqs[0].length;
  const inputs = new Int32Array(batchSize * T);
  const targets = new Int32Array(batchSize * T);
  for (let r = 0; r < batchSize; r++) {
    const p = seqs[Math.floor(rng() * seqs.length)];
    inputs.set([bosId, ...p.slice(0, T - 1)], r * T);
    targets.set(p, r * T);
  }
  return { inputs, targets };
}

/** 字表指纹：用 \u0000 连接后做 FNV-1a 32 位（按 UTF-16 码元），8 位小写十六进制 */
export function vocabHash(vocab: string[]): string {
  const s = vocab.join('\u0000');
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(16).padStart(8, '0');
}

/**
 * 不能用的数据（HTTP 200 但内容不对）在这里就抛错，页面才会显示「重试」，而不是建模型时才出错：
 * 至少 2 首（训练集、验证集各至少一首），每首都是五绝，预设诗的字都在字表里
 */
export function createDataset(file: PoemsFile): Dataset {
  const poems: unknown = file?.poems;
  if (!Array.isArray(poems) || poems.length < 2) throw new Error('数据集至少要有 2 首诗');
  const bad = poems.findIndex((p) => typeof p !== 'string' || !isWujue(p));
  if (bad >= 0) throw new Error(`数据集第 ${bad + 1} 首不是五言绝句`);
  const vocab = buildVocab(file.poems);
  const stoi = new Map(vocab.map((c, i) => [c, i]));
  for (const { poem } of presets) encode(poem, stoi); // 字表外的字会抛「字符不在字表中」
  const { train, val } = splitTrainVal(file.poems.map((p) => encode(p, stoi)));
  return { poems: file.poems, vocab, stoi, bosId: vocab.length - 1, train, val, hash: vocabHash(vocab) };
}

export async function loadDataset(url = 'data/poems.json', fetchFn: typeof fetch = fetch): Promise<Dataset> {
  const res = await fetchFn(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return createDataset(await res.json());
}

/** 自定义诗：去掉空白和标点 → 报字表外的字（去重、按出现顺序）→ 要恰好 20 字，通过后补上「，。，。」 */
export function validateCustomPoem(
  raw: string,
  stoi: Map<string, number>,
): { ok: true; poem: string } | { ok: false; error: string } {
  const chars = [...raw.replace(/[\s\p{P}]/gu, '')];
  const unknown = [...new Set(chars.filter((c) => !stoi.has(c)))];
  if (unknown.length) return { ok: false, error: `以下字符不在字表中：${unknown.join('、')}` };
  if (chars.length !== 20) return { ok: false, error: `需要恰好 20 个汉字（当前 ${chars.length} 个）` };
  return { ok: true, poem: chars.map((c, i) => c + (i % 10 === 4 ? '，' : i % 10 === 9 ? '。' : '')).join('') };
}
