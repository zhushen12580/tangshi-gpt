import { expect, test } from 'vitest';
import { inkColor } from '../../src/ui/heatmap';

test('注意力权重在线性宣纸色阶内取色', () => {
  expect(inkColor(0)).toBe('rgb(245, 239, 224)');
  expect(inkColor(0.5)).toBe('rgb(138, 133, 124)');
  expect(inkColor(1)).toBe('rgb(31, 26, 23)');
  expect(inkColor(2)).toBe(inkColor(1));
  expect(inkColor(-1)).toBe(inkColor(0));
});
