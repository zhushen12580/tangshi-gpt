import { MODEL_DEFAULTS } from '../model';

export function inkColor(w: number): string {
  const weight = Math.min(1, Math.max(0, w));
  const channels = [245, 239, 224].map((paper, i) => Math.round(paper + ([31, 26, 23][i] - paper) * weight));
  return `rgb(${channels.join(', ')})`;
}

export interface Heatmap {
  update(maps: number[][][][], labels: string[], step: number): void;
  select(layer: number, head: number): void;
}

type Cell = { row: number; col: number };

function draw(canvas: HTMLCanvasElement, weights: number[][], cell?: Cell): void {
  const width = Math.round(canvas.getBoundingClientRect().width * devicePixelRatio);
  if (!width) return;
  if (canvas.width !== width || canvas.height !== width) {
    canvas.width = canvas.height = width;
  }
  const ctx = canvas.getContext('2d')!;
  const boundary = (i: number) => Math.round(i * width / weights.length);
  ctx.fillStyle = inkColor(0);
  ctx.fillRect(0, 0, width, width);
  weights.forEach((row, i) => {
    for (let j = 0; j <= i; j++) {
      ctx.fillStyle = inkColor(row[j]);
      ctx.fillRect(boundary(j), boundary(i), boundary(j + 1) - boundary(j), boundary(i + 1) - boundary(i));
    }
  });
  if (cell) {
    // 朱砂描边画在格子里面，不盖住相邻格
    const line = 2 * devicePixelRatio;
    const x = boundary(cell.col);
    const y = boundary(cell.row);
    ctx.strokeStyle = '#c0392b';
    ctx.lineWidth = line;
    ctx.strokeRect(x + line / 2, y + line / 2, boundary(cell.col + 1) - x - line, boundary(cell.row + 1) - y - line);
  }
}

export function createHeatmap(root: HTMLElement): Heatmap {
  const { nLayer, nHead } = MODEL_DEFAULTS;
  const stepLabel = document.createElement('p');
  stepLabel.dataset.testid = 'heatmap-step';
  const thumbs = document.createElement('div');
  thumbs.className = 'heatmap-thumbs';
  const layout = document.createElement('div');
  layout.className = 'heatmap-layout';
  const axisX = document.createElement('div');
  axisX.dataset.testid = 'heatmap-axis-x';
  const axisY = document.createElement('div');
  axisY.dataset.testid = 'heatmap-axis-y';
  const main = document.createElement('canvas');
  main.dataset.testid = 'heatmap-main';
  main.tabIndex = 0;
  main.setAttribute('aria-label', '注意力放大图');
  const readout = document.createElement('p');
  readout.dataset.testid = 'heatmap-readout';
  layout.append(axisX, axisY, main);
  root.replaceChildren(stepLabel, thumbs, layout, readout); // 换掉 HTML 里「数据加载后显示」的空状态

  let layer = 0;
  let head = 0;
  let maps: number[][][][] | undefined;
  let labels: string[] = [];
  let cell: Cell | undefined;
  const drawMain = () => {
    if (!maps) return;
    const weights = maps[layer][head];
    draw(main, weights, cell);
    if (cell) {
      readout.textContent = `「${labels[cell.row]}」看「${labels[cell.col]}」：${weights[cell.row][cell.col].toFixed(2)}`;
    }
  };
  // 高亮格只在下三角：行夹在 0–23，列夹在 0–行
  const moveTo = (row: number, col: number) => {
    const last = labels.length - 1;
    const r = Math.min(last, Math.max(0, row));
    cell = { row: r, col: Math.min(r, Math.max(0, col)) };
    drawMain();
  };
  const ARROWS: Record<string, Cell> = {
    ArrowUp: { row: -1, col: 0 },
    ArrowDown: { row: 1, col: 0 },
    ArrowLeft: { row: 0, col: -1 },
    ArrowRight: { row: 0, col: 1 },
  };
  main.addEventListener('focus', () => {
    if (!cell && maps) moveTo(0, 0);
  });
  main.addEventListener('keydown', (e) => {
    const d = ARROWS[e.key];
    if (!d || !maps) return;
    e.preventDefault(); // 不让方向键滚动页面
    moveTo((cell?.row ?? 0) + d.row, (cell?.col ?? 0) + d.col);
  });
  main.addEventListener('mousemove', (e) => {
    if (!maps) return;
    const rect = main.getBoundingClientRect();
    const n = labels.length;
    moveTo(Math.floor((e.clientY - rect.top) / rect.height * n), Math.floor((e.clientX - rect.left) / rect.width * n));
  });
  const buttons = Array.from({ length: nLayer * nHead }, (_, i) => {
    const l = Math.floor(i / nHead);
    const h = i % nHead;
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.testid = 'heatmap-thumb';
    button.setAttribute('aria-pressed', String(i === 0));
    const canvas = document.createElement('canvas');
    button.append(canvas, `第${l + 1}层·头${h + 1}`);
    button.addEventListener('click', () => select(l, h));
    thumbs.append(button);
    return { button, canvas, l, h };
  });
  new ResizeObserver(drawMain).observe(main);

  function select(l: number, h: number): void {
    buttons[layer * nHead + head].button.setAttribute('aria-pressed', 'false');
    layer = l;
    head = h;
    buttons[layer * nHead + head].button.setAttribute('aria-pressed', 'true');
    drawMain();
  }

  return {
    select,
    update(nextMaps, nextLabels, step) {
      maps = nextMaps;
      labels = nextLabels;
      for (const axis of [axisX, axisY]) {
        axis.replaceChildren(...labels.map((label) => {
          const span = document.createElement('span');
          span.textContent = label;
          return span;
        }));
      }
      for (const { canvas, l, h } of buttons) draw(canvas, maps[l][h]);
      drawMain();
      stepLabel.textContent = `来自第 ${step} 步的模型`;
    },
  };
}
