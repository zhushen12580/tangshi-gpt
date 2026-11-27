import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { FPS, buildTimeline, formatPercent, traceErrors, transcript, type Caption, type Timeline, type Trace } from '../../explainer/timeline';
import { guessCount } from '../../src/explain';
import presets from '../../src/presets.json';

// 视频和文字版里的数字只能从入库的 trace 和代码来；改了 trace 要重新渲染（npm run build:explainer-video）
const trace: Trace = JSON.parse(readFileSync('explainer/trace.json', 'utf8'));
const tl = buildTimeline(trace);
const scene = (name: string, t = tl) => t.scenes.find((s) => s.name === name)!.captions.map((c) => c.text).join('\n');
const pct = formatPercent;
const nll = (p: number) => (-Math.log(p)).toFixed(2);

/** 渲染时第 i 帧画的是 i / FPS 秒（explainer/render.ts）：数出这条字幕实际出现在多少帧上 */
const framesShown = (c: Caption, duration: number) =>
  Array.from({ length: Math.round(duration * FPS) }, (_, i) => i / FPS).filter((time) => time >= c.start && time < c.end).length;

/** 每条字幕显示 ≥ max(2, 字数 / 8) 秒（规格下限），按 30 fps 数帧也够，依次出现，总长 60–90 秒 */
function expectReadable(t: Timeline) {
  let last = 0;
  for (const s of t.scenes) {
    expect(s.start).toBeGreaterThanOrEqual(last);
    for (const c of s.captions) {
      const min = Math.max(2, [...c.text].length / 8);
      expect(c.end - c.start).toBeGreaterThanOrEqual(min - 1e-9); // 两个「帧号 / FPS」相减有浮点误差；下面按帧数是精确的
      // 只看秒数不够：按每秒 8 字排时一个字 3.75 帧，以前采样会少 1 帧（verify 发现 F-a74aa2f447ad）
      expect(framesShown(c, t.duration) / FPS, c.text).toBeGreaterThanOrEqual(min);
      expect(c.start).toBeGreaterThanOrEqual(last);
      last = c.end;
    }
    expect(s.end).toBe(last);
  }
  expect(t.duration).toBe(last);
  expect(t.duration).toBeGreaterThanOrEqual(60);
  expect(t.duration).toBeLessThanOrEqual(90);
}

/**
 * review 发现 F-b4ea32f7f59f 的 trace：真实 trace 的副本，第 50–250 步的样本换成数据集里第一首含「月」「秋风」「江」「花」「酒」的诗，
 * 第 1500 步的样本是训练集原文，验证 loss 最低点挪到第 1050 步（5.3）。意象全齐、多了背诗句，字幕按每秒 7.5 字排是 90.1 秒
 */
function longestTrace(): Trace {
  const poems: string[] = JSON.parse(readFileSync('public/data/poems.json', 'utf8')).poems;
  const inTrain = (text: string) => poems.indexOf(text) % 20 !== 0; // 与 splitTrainVal 一致：每 20 首的第 1 首进验证集
  const samples = trace.samples.map((s) => ({ ...s }));
  ['月', '秋风', '江', '花', '酒'].forEach((m, i) => {
    const text = poems.find((p) => p.includes(m))!;
    samples[i + 1] = { ...samples[i + 1], text, inTrain: inTrain(text) };
  });
  samples[30] = { ...samples[30], text: poems[1], inTrain: true };
  const val = trace.val.map((v) => (v.step === 1050 ? { ...v, loss: 5.3 } : v));
  samples[21] = { ...samples[21], val: 5.3 };
  return { ...trace, samples, val, best: val[21] };
}

describe('buildTimeline', () => {
  test('分镜依次为片头、7 个原理分镜、片尾', () => {
    expect(tl.scenes.map((s) => s.name)).toEqual(
      ['片头', '猜下一个字', '对答案、扣分', '微调参数', 'loss 下降', '训练 loss 与验证 loss', '温度', '注意力', '片尾']);
    expect(tl.scenes.map((s) => s.id)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  test('每条字幕显示 ≥ max(2, 字数 / 8) 秒，依次出现，总长 60–90 秒', () => {
    expectReadable(tl);
  });

  test('字幕多到按每秒 7.5 字排会超过 90 秒时，按规格下限每秒 8 字排，仍在 60–90 秒内', () => {
    const long = longestTrace();
    expect(traceErrors(long)).toEqual([]);
    const t = buildTimeline(long);
    expect(scene('loss 下降', t)).toContain('月、秋风、江、花、酒开始出现');
    expect(scene('训练 loss 与验证 loss', t)).toContain('第 1500 步写出的那首，就是训练集里的原文。');
    expectReadable(t);
  });

  test('按每秒 8 字排也超过 90 秒时报错，不出超长的视频', () => {
    const tooLong: Trace = { ...trace, guess: { ...trace.guess, answer: '河'.repeat(200) } }; // 答案在字幕里出现两次
    expect(() => buildTimeline(tooLong)).toThrow(/超过 90 秒/);
  });

  test('字幕里没有没填上的模板', () => {
    for (const s of tl.scenes) for (const c of s.captions) expect(c.text).not.toMatch(/[{}]|undefined|NaN/);
  });

  test('片头写明后端和步数', () => {
    expect(scene('片头')).toContain({ webgpu: 'WebGPU', webgl: 'WebGL' }[trace.meta.backend]);
    expect(scene('片头')).toContain('1500 步');
  });

  test('对答案：「河」在第 0 步和最好模型下的概率与扣分', () => {
    const { step0, best } = trace.guess;
    const text = scene('对答案、扣分');
    for (const s of ['「河」', pct(step0.p), nll(step0.p), pct(best.p), nll(best.p), `第 ${trace.best.step} 步`]) expect(text).toContain(s);
  });

  test('对答案：措辞跟着实测的方向走，概率升、降、持平都写对', () => {
    const { step0 } = trace.guess;
    const withBest = (p: number) => scene('对答案、扣分', buildTimeline({ ...trace, guess: { ...trace.guess, best: { ...trace.guess.best, p } } }));
    const up = withBest(step0.p * 10);
    expect(up).toContain(`概率升到 ${pct(step0.p * 10)}，只扣 ${nll(step0.p * 10)}`);

    // 最好的模型是按整套验证集选的，这一个字的概率可能反而降了（review 发现 F-f9e0582eeb7d）
    const down = withBest(step0.p / 2);
    expect(down).toContain(`概率反而降到 ${pct(step0.p / 2)}，要扣 ${nll(step0.p / 2)}`);
    expect(down).not.toMatch(/升到|只扣/);

    const flat = withBest(step0.p);
    expect(flat).toContain(`概率还是 ${pct(step0.p)}，扣分还是 ${nll(step0.p)}`);
    expect(flat).not.toMatch(/升到|降到|只扣/);
  });

  test('微调参数：参数量和 batch', () => {
    const text = scene('微调参数');
    for (const s of ['106 万', '1059584', `${trace.meta.batch} 首`, `${trace.meta.steps} 步`]) expect(text).toContain(s);
  });

  test('loss 下降：字表大小、ln V、最好模型相当于在几个字里蒙', () => {
    const text = scene('loss 下降');
    for (const s of [`${trace.meta.vocab}`, trace.meta.lnVocab.toFixed(2), `${guessCount(trace.best.loss)} 个字`]) expect(text).toContain(s);
  });

  test('loss 下降：意象只算最好模型之前（含）的真实样本', () => {
    const motifs = ['月', '秋风', '江', '花', '酒'].filter((m) => trace.samples.some((s) => s.step <= trace.best.step && s.text.includes(m)));
    expect(scene('loss 下降')).toContain(motifs.length ? `${motifs.join('、')}开始出现` : '常用字开始出现');

    const withText = (before: string, after: string): Trace =>
      ({ ...trace, samples: trace.samples.map((s) => ({ ...s, text: s.step <= trace.best.step ? before : after })) });
    const moon = scene('loss 下降', buildTimeline(withText('明月照山间，清泉石上流。空山新雨后，天气晚来秋。', '劝君更尽一杯酒，西出阳关。')));
    expect(moon).toContain('月开始出现');
    expect(moon).not.toContain('酒');
    const none = scene('loss 下降', buildTimeline(withText('白日依山尽，黄河入海流。欲穷千里目，更上一层楼。', '酒')));
    expect(none).toContain('常用字开始出现');
    expect(none).not.toContain('酒');
  });

  test('训练 loss 与验证 loss：模拟考的首数、最低点、回升', () => {
    const text = scene('训练 loss 与验证 loss');
    for (const s of [`${trace.meta.valCount} 首`, trace.loss.at(-1)!.toFixed(2), trace.best.loss.toFixed(2), trace.val.at(-1)!.loss.toFixed(2), '它开始背诗了（过拟合）'])
      expect(text).toContain(s);
  });

  test('背诗句只在最好模型之后真有训练集原文时才写', () => {
    const copyAfter = trace.samples.find((s) => s.step > trace.best.step && s.inTrain);
    const text = scene('训练 loss 与验证 loss');
    if (copyAfter) expect(text).toContain(`第 ${copyAfter.step} 步写出的那首，就是训练集里的原文。`);
    else expect(text).not.toContain('就是训练集里的原文');

    const mark = (pred: (step: number) => boolean): Trace => ({ ...trace, samples: trace.samples.map((s) => ({ ...s, inTrain: pred(s.step) })) });
    expect(scene('训练 loss 与验证 loss', buildTimeline(mark((st) => st === 1500)))).toContain('第 1500 步写出的那首，就是训练集里的原文。');
    expect(scene('训练 loss 与验证 loss', buildTimeline(mark((st) => st <= trace.best.step)))).not.toContain('就是训练集里的原文');
  });

  test('温度：两组温度都来自 trace', () => {
    for (const { temperature } of trace.temperature) expect(scene('温度')).toContain(`温度 ${temperature}`);
  });

  test('注意力：「海」那一行在全部层、头、第 1–9 列里最大的权重', () => {
    const tokens = ['<s>', ...presets.find((p) => p.title === '登鹳雀楼')!.poem];
    let peak = { layer: 0, head: 0, col: 1, w: -1 };
    trace.attention.forEach((layer, l) => layer.forEach((head, h) => {
      for (let col = 1; col <= 9; col++) if (head[10][col] > peak.w) peak = { layer: l, head: h, col, w: head[10][col] };
    }));
    expect(tokens[10]).toBe('海');
    expect(scene('注意力')).toContain(`第 ${peak.layer + 1} 层第 ${peak.head + 1} 个头在「海」这个位置最看重「${tokens[peak.col]}」（${peak.w.toFixed(2)}）`);
  });

  test('片尾是网址', () => {
    expect(scene('片尾')).toContain('zhushen12580.github.io/tangshi-gpt');
  });
});

test('formatPercent：≥ 10% 写整数，否则两位有效数字', () => {
  expect(formatPercent(0.5)).toBe('50%');
  expect(formatPercent(0.0123)).toBe('1.2%');
  expect(formatPercent(0.00123)).toBe('0.12%');
  expect(formatPercent(0.000359)).toBe('0.036%');
});

test('transcript：片名加每个分镜的名称和字幕', () => {
  const text = transcript(trace);
  expect(text.startsWith('唐诗 GPT')).toBe(true);
  for (const s of tl.scenes) {
    expect(text).toContain(s.name);
    for (const c of s.captions) expect(text).toContain(c.text);
  }
});

describe('入库的视频和文字版', () => {
  test('explainer.txt 以 BOM 开头，其余与 trace 现生成的文字版逐字一致', () => {
    const file = readFileSync('public/explainer.txt', 'utf8');
    expect(file.startsWith('﻿')).toBe(true);
    expect(file.slice(1)).toBe(transcript(trace));
  });

  test('explainer.mp4 ≤ 4 MiB，H.264，moov 在 mdat 前，时长 60–90 秒', () => {
    const buf = readFileSync('public/explainer.mp4');
    expect(buf.length).toBeLessThanOrEqual(4 * 1024 * 1024);
    const boxes = (start: number, end: number) => {
      const out: { type: string; start: number; end: number }[] = [];
      for (let at = start; at < end;) {
        let size = buf.readUInt32BE(at);
        if (size === 1) size = Number(buf.readBigUInt64BE(at + 8));
        if (size === 0) size = end - at;
        out.push({ type: buf.toString('latin1', at + 4, at + 8), start: at, end: at + size });
        at += size;
      }
      return out;
    };
    const top = boxes(0, buf.length);
    const types = top.map((b) => b.type);
    expect(types[0]).toBe('ftyp');
    expect(types).toContain('mdat');
    expect(types.indexOf('moov')).toBeGreaterThanOrEqual(0);
    expect(types.indexOf('moov')).toBeLessThan(types.indexOf('mdat'));
    expect(buf.includes('avc1', 0, 'latin1')).toBe(true);

    const moov = top.find((b) => b.type === 'moov')!;
    const mvhd = boxes(moov.start + 8, moov.end).find((b) => b.type === 'mvhd')!;
    const v1 = buf[mvhd.start + 8] === 1; // version 1 的时间字段是 64 位
    const timescale = buf.readUInt32BE(mvhd.start + (v1 ? 28 : 20));
    const duration = v1 ? Number(buf.readBigUInt64BE(mvhd.start + 32)) : buf.readUInt32BE(mvhd.start + 24);
    expect(duration / timescale).toBeGreaterThanOrEqual(60);
    expect(duration / timescale).toBeLessThanOrEqual(90);
  });
});
