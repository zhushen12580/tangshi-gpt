import { expect, test } from 'vitest';
import { emaSeries } from '../../src/ui/lossChart';

test('emaSeries：空数组为空，第一个值原样，之后 0.9·前值 + 0.1·当前值', () => {
  expect(emaSeries([])).toEqual([]);
  const s = emaSeries([1, 2]);
  expect(s).toHaveLength(2);
  expect(s[0]).toBe(1);
  expect(s[1]).toBeCloseTo(1.1, 10);
});
