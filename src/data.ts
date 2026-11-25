// 数据：字表、编码、训练/验证切分、取 batch、加载 poems.json

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

export function createDataset(file: PoemsFile): Dataset {
  const vocab = buildVocab(file.poems);
  const stoi = new Map(vocab.map((c, i) => [c, i]));
  const { train, val } = splitTrainVal(file.poems.map((p) => encode(p, stoi)));
  return { poems: file.poems, vocab, stoi, bosId: vocab.length - 1, train, val, hash: vocabHash(vocab) };
}

export async function loadDataset(url = 'data/poems.json', fetchFn: typeof fetch = fetch): Promise<Dataset> {
  const res = await fetchFn(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return createDataset(await res.json());
}
