// 原理视频的画面：画出时间点 time 的那一帧。纯绘制，只用传入的参数，同一时间点画出的结果相同（不用随机数和时钟）

import { inkColor } from '../src/ui/heatmap';
import { emaSeries } from '../src/ui/lossChart';
import { PEAK_ROW, TRANSITION, attentionPeak, attentionTokens, formatPercent, type Scene, type Timeline, type Trace } from './timeline';

export const WIDTH = 1280;
export const HEIGHT = 720;
const FONT = '"Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", "SimSun", serif'; // 同 src/style.css 的 --font-serif
const PAPER = '#f5efe0'; // 宣纸
const INK = '#1f1a17'; // 墨
const CINNABAR = '#c0392b'; // 朱砂
const FAINT = 'rgba(31, 26, 23, 0.3)';
const CAPTION = { size: 36, lineHeight: 50, lastLine: 672, maxWidth: 1120 }; // 字幕在画面下方，画面内容都在 y < 585
const NO_LINE_START = '，。、；：）」”！？'; // 这些标点不放在行首
const TOKEN = /[（「“]*(?:[A-Za-z0-9.%<>/_-]+|.)/gu; // 折行的最小单位：数字、英文和网址不拆开，开括号跟着后面的字

type G = CanvasRenderingContext2D;

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const ease = (x: number) => clamp01(x) * clamp01(x) * (3 - 2 * clamp01(x));
const between = (time: number, a: number, b: number) => clamp01((time - a) / (b - a));
const hash = (i: number) => { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x); }; // 固定的伪随机，[0, 1)
const f2 = (x: number) => x.toFixed(2);

function text(g: G, s: string, x: number, y: number, size: number, color = INK, align: CanvasTextAlign = 'left') {
  g.font = `${size}px ${FONT}`;
  g.fillStyle = color;
  g.textAlign = align;
  g.fillText(s, x, y);
}

/** 一行概率条：右对齐的字、条、数值。scale 是数值 1 对应的像素宽 */
function bar(g: G, label: string, value: number, shown: string, x: number, y: number, scale: number, color = INK, size = 30) {
  text(g, label, x, y, size, color, 'right');
  const w = Math.max(3, value * scale);
  g.fillStyle = color;
  g.fillRect(x + 14, y - 11, w, 22);
  text(g, shown, x + 26 + w, y, 22, color);
}

/** 一个记号框，比如 <s> 或者待猜的「？」 */
function tag(g: G, s: string, x: number, y: number, size: number, color: string) {
  g.font = `${size}px ${FONT}`;
  const w = g.measureText(s).width + size * 0.5;
  g.strokeStyle = color;
  g.lineWidth = 2;
  g.strokeRect(x, y - size * 0.65, w, size * 1.3);
  text(g, s, x + w / 2, y, size, color, 'center');
  return w;
}

/** 按宽度折行；句末标点跟着上一行 */
function wrap(g: G, s: string, maxWidth: number): string[] {
  const lines = [''];
  for (const token of s.match(TOKEN) ?? []) {
    const line = lines[lines.length - 1];
    if (line && g.measureText(line + token).width > maxWidth && !NO_LINE_START.includes(token)) lines.push(token.trimStart());
    else lines[lines.length - 1] = line + token;
  }
  return lines;
}

/** 行数不变的前提下尽量让各行一样长，避免第二行只剩一两个字 */
function balancedWrap(g: G, s: string, maxWidth: number): string[] {
  const n = wrap(g, s, maxWidth).length;
  let width = g.measureText(s).width / n;
  while (wrap(g, s, width).length > n) width += 10;
  return wrap(g, s, width);
}

function title(g: G) {
  text(g, '唐诗 GPT', WIDTH / 2, 250, 104, INK, 'center');
  g.fillStyle = CINNABAR;
  g.fillRect(WIDTH / 2 - 150, 330, 300, 4);
  text(g, '1 分钟看懂它怎么学会写诗', WIDTH / 2, 400, 48, INK, 'center');
}

/** 分镜 1：从 <s> 起逐字写出 demo，右边是正在猜的那个位置的前 3 个候选 */
function guessNext(g: G, t: Trace, s: Scene, time: number) {
  const chars = [...t.demo.text];
  const written = Math.floor(ease(between(time, s.start + 0.3, s.end - 1)) * chars.length);
  tag(g, '<s>', 120, 150, 34, CINNABAR);
  chars.forEach((c, i) => {
    const x = 150 + (i % 6) * 68;
    const y = 240 + Math.floor(i / 6) * 84;
    if (i < written) text(g, c, x, y, 54, INK, 'center');
    else if (i === written) tag(g, '？', x - 30, y, 40, CINNABAR);
  });
  const pos = Math.min(written, chars.length - 1);
  text(g, `第 ${pos + 1} 个字的候选：`, 640, 160, 30); // 标点位置被格式规定死，只有 1 个候选
  t.demo.top[pos].forEach((c, r) => {
    const picked = c.char === chars[pos];
    bar(g, c.char, c.p, formatPercent(c.p), 700, 240 + r * 80, 400, picked ? CINNABAR : INK, 40);
  });
  text(g, `按概率抽一个：这次抽中「${chars[pos]}」`, 640, 480, 30, CINNABAR);
}

/** 分镜 2：「白日依山尽，黄」后面的前 5 个候选，答案「河」用朱砂色，第 0 步和最好模型对比 */
function answer(g: G, t: Trace, s: Scene, time: number) {
  const { guess } = t;
  let x = 120 + tag(g, '<s>', 120, 140, 34, INK) + 16;
  for (const c of guess.context) { text(g, c, x + 26, 140, 52, INK, 'center'); x += 62; }
  tag(g, guess.answer, x, 140, 52, CINNABAR);
  const panels = [{ name: '第 0 步（参数还是随机的）', g: guess.step0, at: s.captions[1].start },
    { name: `第 ${t.best.step} 步（最好的模型）`, g: guess.best, at: s.captions[2].start }];
  const max = Math.max(...panels.flatMap((p) => [p.g.p, ...p.g.top.map((c) => c.p)]));
  panels.forEach((p, k) => {
    if (time < p.at) return;
    const px = 150 + k * 560;
    text(g, p.name, px - 30, 220, 28);
    const rows = p.g.top.some((c) => c.char === guess.answer) ? p.g.top : [...p.g.top, { char: guess.answer, p: p.g.p }];
    rows.forEach((c, r) => {
      const hit = c.char === guess.answer;
      const shown = hit ? `${formatPercent(c.p)} · 扣 ${f2(-Math.log(c.p))}` : formatPercent(c.p);
      bar(g, c.char, c.p, shown, px, 276 + r * 50 + (r === 5 ? 10 : 0), 260 / max, hit ? CINNABAR : INK);
    });
  });
}

/** 分镜 3：一大片代表参数的小点，一起往同一个方向挪，步数走到 steps */
function params(g: G, t: Trace, s: Scene, time: number) {
  const p = ease(between(time, s.start + 0.3, s.end - 0.8));
  const cols = 40;
  const rows = 12;
  for (let i = 0; i < cols * rows; i++) {
    const x = 140 + ((i % cols) + 0.5) * 21 + (hash(i) - 0.5) * 70 * (1 - p) + p * 30;
    const y = 180 + (Math.floor(i / cols) + 0.5) * 30 + (hash(i + 7919) - 0.5) * 40 * (1 - p);
    g.fillStyle = hash(i + 104729) < 0.5 + p / 2 ? INK : FAINT;
    g.beginPath();
    g.arc(x, y, 4.5, 0, Math.PI * 2);
    g.fill();
  }
  g.strokeStyle = g.fillStyle = CINNABAR;
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(1040, 360);
  g.lineTo(1180, 360);
  g.stroke();
  g.beginPath();
  g.moveTo(1196, 360);
  g.lineTo(1170, 344);
  g.lineTo(1170, 376);
  g.fill();
  text(g, '猜对', 1110, 320, 34, CINNABAR, 'center');
  text(g, `${t.meta.params} 个参数`, 140, 130, 30);
  text(g, `第 ${Math.round(p * t.meta.steps)} / ${t.meta.steps} 步`, 1200, 130, 40, CINNABAR, 'right');
  text(g, `每步 ${t.meta.batch} 首诗`, 1200, 560, 26, INK, 'right');
}

/** loss 曲线：训练 loss 细线 + EMA 粗线，可选验证 loss 的朱砂点；横轴 0–xMax 步，只画到 upTo 步 */
function lossChart(g: G, t: Trace, box: { x: number; y: number; w: number; h: number }, xMax: number, upTo: number, withVal: boolean) {
  const yMax = Math.ceil(Math.max(t.meta.lnVocab, ...t.loss, ...t.val.map((v) => v.loss)));
  const X = (step: number) => box.x + (step / xMax) * box.w;
  const Y = (v: number) => box.y + box.h - (v / yMax) * box.h;
  g.strokeStyle = INK;
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(box.x, box.y);
  g.lineTo(box.x, box.y + box.h);
  g.lineTo(box.x + box.w, box.y + box.h);
  g.stroke();
  for (let v = 0; v <= yMax; v += 2) text(g, `${v}`, box.x - 12, Y(v), 22, INK, 'right');
  text(g, '0', box.x, box.y + box.h + 22, 22, INK, 'center');
  text(g, `${xMax / 2}`, X(xMax / 2), box.y + box.h + 22, 22, INK, 'center');
  text(g, `第 ${xMax} 步`, box.x + box.w, box.y + box.h + 22, 22, INK, 'right');
  g.setLineDash([8, 8]);
  g.strokeStyle = FAINT;
  g.beginPath();
  g.moveTo(box.x, Y(t.meta.lnVocab));
  g.lineTo(box.x + box.w, Y(t.meta.lnVocab));
  g.stroke();
  g.setLineDash([]);
  text(g, `瞎蒙：ln ${t.meta.vocab} ≈ ${f2(t.meta.lnVocab)}`, box.x + 40, Y(t.meta.lnVocab) - 20, 22); // 放左边：右边会被回升的验证 loss 穿过
  // loss[i] 是第 i + 1 步的训练 loss
  const line = (values: number[], color: string, width: number) => {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.beginPath();
    values.slice(0, upTo).forEach((v, i) => (i ? g.lineTo(X(i + 1), Y(v)) : g.moveTo(X(i + 1), Y(v))));
    g.stroke();
  };
  line(t.loss, FAINT, 1.5);
  line(emaSeries(t.loss), INK, 3.5);
  if (withVal) {
    const pts = t.val.filter((v) => v.step <= upTo);
    g.strokeStyle = g.fillStyle = CINNABAR;
    pts.forEach((v, i) => {
      if (i) {
        g.lineWidth = v.step > t.best.step ? 4 : 2; // 最低点之后的回升段加粗
        g.beginPath();
        g.moveTo(X(pts[i - 1].step), Y(pts[i - 1].loss));
        g.lineTo(X(v.step), Y(v.loss));
        g.stroke();
      }
      g.beginPath();
      g.arc(X(v.step), Y(v.loss), 6, 0, Math.PI * 2);
      g.fill();
    });
  }
  return { X, Y };
}

/** 分镜 4：loss 曲线画到最好模型那一步，右边依次出现第 0 步、中间一步、最好那一步的样本 */
function lossDown(g: G, t: Trace, s: Scene, time: number) {
  const { best } = t;
  const upTo = Math.round(ease(between(time, s.start + 0.3, s.captions[1].start + 1)) * best.step);
  const { X, Y } = lossChart(g, t, { x: 110, y: 120, w: 600, h: 420 }, best.step, upTo, false);
  if (time >= s.captions[1].start) {
    g.fillStyle = CINNABAR;
    g.beginPath();
    g.arc(X(best.step), Y(best.loss), 8, 0, Math.PI * 2);
    g.fill();
    text(g, `验证 loss ${f2(best.loss)}`, X(best.step) - 14, Y(best.loss) - 36, 26, CINNABAR, 'right');
  }
  const mid = t.samples.reduce((a, b) => (Math.abs(b.step - best.step / 2) < Math.abs(a.step - best.step / 2) ? b : a));
  const picks = [...new Map([t.samples[0], mid, t.samples.find((x) => x.step === best.step)!].map((x) => [x.step, x])).values()];
  picks.forEach((x, i) => {
    if (upTo < x.step) return;
    const y = 140 + i * 145;
    text(g, `第 ${x.step} 步写的：`, 800, y, 26, CINNABAR);
    const chars = [...x.text];
    text(g, chars.slice(0, 12).join(''), 800, y + 44, 32);
    text(g, chars.slice(12).join(''), 800, y + 86, 32);
  });
}

/** 分镜 5：训练 loss 一路画到最后，验证 loss 的点在最低点之后回升 */
function trainVsVal(g: G, t: Trace, s: Scene, time: number) {
  const { best, meta } = t;
  const upTo = Math.round(best.step + ease(between(time, s.captions[1].start, s.captions[2].start - 0.5)) * (meta.steps - best.step));
  const { X, Y } = lossChart(g, t, { x: 110, y: 140, w: 1040, h: 400 }, meta.steps, upTo, true);
  g.fillStyle = INK;
  g.fillRect(700, 88, 40, 4);
  text(g, '训练 loss', 752, 90, 26);
  g.fillStyle = CINNABAR;
  g.beginPath();
  g.arc(920, 90, 7, 0, Math.PI * 2);
  g.fill();
  text(g, '验证 loss（模拟考）', 938, 90, 26, CINNABAR);
  text(g, `最低 ${f2(best.loss)}`, X(best.step), Y(best.loss) - 34, 26, CINNABAR, 'center');
  if (upTo === meta.steps) {
    const last = t.val[t.val.length - 1];
    text(g, `回升到 ${f2(last.loss)}`, X(last.step) - 14, Y(last.loss) - 26, 26, CINNABAR, 'right');
    text(g, `训练 ${f2(t.loss[t.loss.length - 1])}`, X(meta.steps) - 14, Y(t.loss[t.loss.length - 1]) - 30, 26, INK, 'right');
  }
}

/** 分镜 6：第 1 个字在温度 0.3 和 1.5 下的前 8 个候选和「其余」 */
function temperature(g: G, t: Trace, s: Scene, time: number) {
  const grow = ease(between(time, s.start, s.start + 0.8));
  t.temperature.forEach((e, k) => {
    const x = 180 + k * 580;
    text(g, `温度 ${e.temperature}`, x - 60, 130, 36, CINNABAR);
    text(g, '第 1 个字的候选', x + 120, 132, 24);
    e.top.forEach((c, r) => bar(g, c.char, c.p * grow, formatPercent(c.p), x, 190 + r * 42, 320));
    bar(g, '其余', e.rest * grow, formatPercent(e.rest), x, 190 + 8 * 42 + 10, 320, CINNABAR, 26);
  });
}

/** 分镜 7：《登鹳雀楼》选中那个头的热力图，「海」那一行和最看重的格子用朱砂框出来 */
function attention(g: G, t: Trace) {
  const peak = attentionPeak(t);
  const head = t.attention[peak.layer][peak.head];
  const tokens = attentionTokens();
  const cell = 19;
  const x0 = 170;
  const y0 = 122;
  tokens.forEach((tok, i) => {
    const size = tok.length > 1 ? 11 : 16;
    text(g, tok, x0 + i * cell + cell / 2, y0 - 12, size, i === peak.col ? CINNABAR : INK, 'center');
    text(g, tok, x0 - 6, y0 + i * cell + cell / 2, size, i === PEAK_ROW ? CINNABAR : INK, 'right');
    head[i].forEach((w, j) => { g.fillStyle = inkColor(w); g.fillRect(x0 + j * cell, y0 + i * cell, cell, cell); });
  });
  g.strokeStyle = CINNABAR;
  g.lineWidth = 2.5;
  g.strokeRect(x0, y0 + PEAK_ROW * cell, tokens.length * cell, cell);
  g.lineWidth = 4;
  g.strokeRect(x0 + peak.col * cell - 2, y0 + PEAK_ROW * cell - 2, cell + 4, cell + 4);
  text(g, `第 ${peak.layer + 1} 层 · 第 ${peak.head + 1} 个头`, 720, 140, 36, CINNABAR);
  text(g, `「${tokens[PEAK_ROW]}」这一行：猜下一个字时看前面各字的权重`, 720, 190, 24);
  for (let col = 1; col < PEAK_ROW; col++) {
    const hit = col === peak.col;
    bar(g, tokens[col], head[PEAK_ROW][col], f2(head[PEAK_ROW][col]), 760, 210 + col * 38, 320, hit ? CINNABAR : INK, 28);
  }
}

function ending(g: G) {
  text(g, '打开页面，亲手训练一次', WIDTH / 2, 280, 60, INK, 'center');
  text(g, 'zhushen12580.github.io/tangshi-gpt', WIDTH / 2, 380, 48, CINNABAR, 'center');
}

const SCENES: ((g: G, t: Trace, s: Scene, time: number) => void)[] =
  [title, guessNext, answer, params, lossDown, trainVsVal, temperature, attention, ending];

/** 画出 time 秒的画面：分镜之间的过渡时间里，下一个分镜从宣纸色淡入 */
export function drawFrame(g: G, t: Trace, tl: Timeline, time: number): void {
  const s = tl.scenes.find((x) => time < x.end) ?? tl.scenes[tl.scenes.length - 1];
  g.save();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = PAPER;
  g.fillRect(0, 0, WIDTH, HEIGHT);
  g.textBaseline = 'middle';
  g.globalAlpha = between(time, s.start - TRANSITION, s.start);
  if (s.id > 0 && s.id < tl.scenes.length - 1) {
    const index = `${s.id} / ${tl.scenes.length - 2}`;
    text(g, index, 64, 56, 28, CINNABAR);
    text(g, s.name, 64 + g.measureText(index).width + 20, 56, 28);
  }
  SCENES[s.id](g, t, s, time);
  const caption = s.captions.find((c) => time >= c.start && time < c.end);
  if (caption) {
    g.font = `${CAPTION.size}px ${FONT}`;
    const lines = balancedWrap(g, caption.text, CAPTION.maxWidth);
    lines.forEach((line, i) => text(g, line, WIDTH / 2, CAPTION.lastLine - (lines.length - 1 - i) * CAPTION.lineHeight, CAPTION.size, INK, 'center'));
  }
  g.restore();
}
