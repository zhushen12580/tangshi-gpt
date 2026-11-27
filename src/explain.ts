// 说明层：把 loss 换成大白话

/** e 的 loss 次方 ≈ 相当于在多少个字里瞎蒙 */
export const guessCount = (x: number): number => Math.max(1, Math.round(Math.exp(x)));

/** 先按 loss-current 的显示取三位小数再算 N，读者从页面上的数字算出来的 N 和这里一致 */
export function guessText(x: number | null): string {
  if (x === null) return '还没开始训练。';
  const s = x.toFixed(3);
  return `现在 ${s} ≈ 在 ${guessCount(Number(s))} 个字里瞎蒙。`;
}
