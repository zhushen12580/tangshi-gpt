// loss 曲线：原始训练 loss（细线）、EMA 平滑（粗线）、验证 loss（朱砂圆点），Canvas 2D 绘制

const PAD = { left: 44, right: 12, top: 10, bottom: 22 }; // 留给坐标刻度的边距（CSS 像素）

/** 指数移动平均：第一个值原样，之后 ema = beta·ema + (1 − beta)·x */
export function emaSeries(values: number[], beta = 0.9): number[] {
  const out: number[] = [];
  values.forEach((v, i) => out.push(i === 0 ? v : beta * out[i - 1] + (1 - beta) * v));
  return out;
}

/** loss[i] 是第 i+1 步的训练 loss；val 是里程碑步的验证 loss（第 0 步在训练之前） */
export function drawLossChart(canvas: HTMLCanvasElement, loss: number[], val: { step: number; loss: number }[]): void {
  // 画布像素按 devicePixelRatio 放大，之后都用 CSS 像素作图，高清屏上不发虚
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext('2d')!;
  g.scale(dpr, dpr);
  if (loss.length + val.length === 0) return;

  const css = getComputedStyle(canvas);
  const ink = css.getPropertyValue('--ink');
  const cinnabar = css.getPropertyValue('--cinnabar');
  const all = [...loss, ...val.map((v) => v.loss)];
  // 不用 Math.min(...all)：训练很久后数组太长，展开参数会爆栈
  const yMin = all.reduce((a, b) => Math.min(a, b));
  const yMax = all.reduce((a, b) => Math.max(a, b));
  const xMax = Math.max(loss.length, val.at(-1)?.step ?? 0, 1);
  const x = (step: number) => PAD.left + (step / xMax) * (w - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (yMax === yMin ? 0.5 : (yMax - v) / (yMax - yMin)) * (h - PAD.top - PAD.bottom);

  // 坐标轴与最小、最大刻度
  g.strokeStyle = g.fillStyle = ink;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(PAD.left, PAD.top);
  g.lineTo(PAD.left, h - PAD.bottom);
  g.lineTo(w - PAD.right, h - PAD.bottom);
  g.stroke();
  g.font = '11px sans-serif';
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  g.fillText(yMax.toFixed(2), PAD.left - 4, y(yMax));
  g.fillText(yMin.toFixed(2), PAD.left - 4, y(yMin));
  g.textBaseline = 'top';
  g.textAlign = 'left';
  g.fillText('0', PAD.left, h - PAD.bottom + 4);
  g.textAlign = 'right';
  g.fillText(String(xMax), w - PAD.right, h - PAD.bottom + 4);

  const line = (values: number[], width: number, alpha: number) => {
    g.globalAlpha = alpha;
    g.lineWidth = width;
    g.beginPath();
    values.forEach((v, i) => g.lineTo(x(i + 1), y(v)));
    g.stroke();
  };
  line(loss, 1, 0.3);
  line(emaSeries(loss), 2.5, 1);

  g.fillStyle = cinnabar;
  for (const v of val) {
    g.beginPath();
    g.arc(x(v.step), y(v.loss), 3.5, 0, 2 * Math.PI);
    g.fill();
  }
}
