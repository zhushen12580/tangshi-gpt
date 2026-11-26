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

function draw(canvas: HTMLCanvasElement, weights: number[][]): void {
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
  root.append(stepLabel, thumbs, layout, readout);

  let layer = 0;
  let head = 0;
  let maps: number[][][][] | undefined;
  const drawMain = () => {
    if (maps) draw(main, maps[layer][head]);
  };
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
    update(nextMaps, labels, step) {
      maps = nextMaps;
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
