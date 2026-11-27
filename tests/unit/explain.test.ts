import { describe, expect, it } from 'vitest';
import { guessCount, guessText, overfitStatus, overfitText } from '../../src/explain';

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

const pts = (...xs: [number, number][]) => xs.map(([step, loss]) => ({ step, loss }));

describe('overfitStatus', () => {
  it('最新一点比最低点高 ≥ 0.1 时报告', () => {
    expect(overfitStatus(pts([0, 8.2], [50, 5.5], [100, 5.7]))).toEqual({ minStep: 50, min: 5.5, latest: 5.7 });
  });

  it('回升不到 0.1、一直下降、只有 1 个点、没有点时不报', () => {
    expect(overfitStatus(pts([0, 8.2], [50, 5.5], [100, 5.59]))).toBeNull();
    expect(overfitStatus(pts([0, 8.2], [50, 5.5], [100, 5.1]))).toBeNull();
    expect(overfitStatus(pts([0, 8.2]))).toBeNull();
    expect(overfitStatus([])).toBeNull();
  });

  it('最低点并列时取最早的', () => {
    expect(overfitStatus(pts([0, 8.2], [50, 5.5], [100, 5.5], [150, 5.7]))).toEqual({ minStep: 50, min: 5.5, latest: 5.7 });
  });

  it('忽略 NaN 和 Infinity', () => {
    expect(overfitStatus(pts([0, 8.2], [50, 5.5], [100, NaN], [150, 5.7]))).toEqual({ minStep: 50, min: 5.5, latest: 5.7 });
    expect(overfitStatus(pts([0, 8.2], [50, 5.5], [100, 5.7], [150, NaN]))).toEqual({ minStep: 50, min: 5.5, latest: 5.7 });
    expect(overfitStatus(pts([0, 8.2], [50, -Infinity], [100, 5.5], [150, Infinity]))).toBeNull();
  });
});

describe('overfitText', () => {
  it('按规格文案，A、B 保留三位小数', () => {
    expect(overfitText({ minStep: 50, min: 5.5, latest: 5.7 })).toBe(
      '它开始背诗了（过拟合）：验证 loss 从第 50 步的最低点 5.500 回升到 5.700。这时写出来的诗，可能就是训练集里的原文。',
    );
  });
});
