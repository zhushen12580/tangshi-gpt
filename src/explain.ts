// 说明层：把 loss 换成大白话

/** e 的 loss 次方 ≈ 相当于在多少个字里瞎蒙 */
export const guessCount = (x: number): number => Math.max(1, Math.round(Math.exp(x)));

/** 先按 loss-current 的显示取三位小数再算 N，读者从页面上的数字算出来的 N 和这里一致 */
export function guessText(x: number | null): string {
  if (x === null) return '还没开始训练。';
  const s = x.toFixed(3);
  return `现在 ${s} ≈ 在 ${guessCount(Number(s))} 个字里瞎蒙。`;
}

/** 验证 loss 比最低点高出这么多才算回升：验证集每次是同一批诗，没有抽样噪声 */
export const OVERFIT_DELTA = 0.1;
/** 浮点减法的误差：5.6 − 5.5 = 0.0999…，恰好回升 0.1 也要算 */
const FLOAT_EPS = 1e-9;

export interface OverfitStatus { minStep: number; min: number; latest: number }

/** 只看有限值：最低点（并列取最早）之后，最新一点高出 ≥ OVERFIT_DELTA 时返回，否则 null */
export function overfitStatus(val: { step: number; loss: number }[]): OverfitStatus | null {
  const finite = val.filter((p) => Number.isFinite(p.loss));
  if (finite.length < 2) return null;
  const best = finite.reduce((a, b) => (b.loss < a.loss ? b : a));
  const latest = finite[finite.length - 1].loss;
  return latest - best.loss >= OVERFIT_DELTA - FLOAT_EPS ? { minStep: best.step, min: best.loss, latest } : null;
}

export const overfitText = (s: OverfitStatus): string =>
  `它开始背诗了（过拟合）：验证 loss 从第 ${s.minStep} 步的最低点 ${s.min.toFixed(3)} 回升到 ${s.latest.toFixed(3)}。这时写出来的诗，可能就是训练集里的原文。`;
