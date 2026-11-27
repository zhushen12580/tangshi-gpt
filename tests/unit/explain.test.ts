import { describe, expect, it } from 'vitest';
import { guessCount, guessText } from '../../src/explain';

describe('guessCount', () => {
  it('e 的 loss 次方取整，至少为 1', () => {
    expect(guessCount(Math.log(3605))).toBe(3605);
    expect(guessCount(0)).toBe(1);
    expect(guessCount(-5)).toBe(1);
  });
});

describe('guessText', () => {
  it('没有 loss 时说还没开始', () => {
    expect(guessText(null)).toBe('还没开始训练。');
  });

  it('N 由显示的三位小数算出', () => {
    expect(guessText(8.19)).toBe('现在 8.190 ≈ 在 3605 个字里瞎蒙。');
    expect(guessText(Math.log(42))).toMatch(/ 在 42 个字里瞎蒙。$/);
  });
});
