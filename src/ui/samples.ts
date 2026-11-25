// 生成记录：每条一首诗，标注步数、验证 loss、采样方式；新的在最上面

import { formatMeta, type SampleRecord } from '../sample';

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ''): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  e.textContent = text;
  return e;
}

function renderItem(r: SampleRecord): HTMLLIElement {
  const li = el('li', 'sample');
  li.dataset.testid = 'sample-item';
  const head = el('p', 'sample-head');
  head.append(el('span', 'sample-meta', formatMeta(r)));
  if (!r.constrained) head.append(el('span', 'format-badge', r.formatOk ? '格式 ✓' : '格式 ✗'));
  const chars = [...r.text];
  const poem = el('div', 'poem');
  for (let i = 0; i < chars.length; i += 6) poem.append(el('div', 'poem-line', chars.slice(i, i + 6).join('')));
  li.append(head, poem);
  return li;
}

export function renderSamples(root: HTMLElement, list: SampleRecord[]): void {
  if (list.length === 0) {
    const empty = el('p', 'empty', '开始训练后，每 50 步会在这里写一首诗');
    empty.dataset.testid = 'samples-empty';
    root.replaceChildren(empty);
    return;
  }
  const ol = el('ol', 'samples');
  ol.append(...list.map(renderItem));
  root.replaceChildren(ol);
}
