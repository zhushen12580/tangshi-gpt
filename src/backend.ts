// 计算后端选择：?backend= 指定起点，按 webgpu → webgl → cpu 逐个尝试

import * as tf from '@tensorflow/tfjs-core';

export type BackendName = 'webgpu' | 'webgl' | 'cpu';
export const BACKEND_LABEL: Record<BackendName, string> = { webgpu: 'WebGPU', webgl: 'WebGL', cpu: 'CPU' };
const ALL: BackendName[] = ['webgpu', 'webgl', 'cpu'];

/** auto（或缺省）从头开始；指定某个后端时从它开始往下；未知值按 auto 处理并提示 */
export function backendOrder(search: string): { order: BackendName[]; notices: string[] } {
  const want = new URLSearchParams(search).get('backend') || 'auto';
  const i = ALL.indexOf(want as BackendName);
  if (i >= 0) return { order: ALL.slice(i), notices: [] };
  return { order: ALL, notices: want === 'auto' ? [] : [`未知后端 ${want}，已自动选择`] };
}

/** 按顺序尝试，第一个成功的就用；tryOne 返回 false 或抛错都算失败 */
export async function chooseBackend(
  order: BackendName[],
  tryOne: (b: BackendName) => Promise<boolean>,
): Promise<{ backend: BackendName; notices: string[] }> {
  for (const b of order) {
    const ok = await tryOne(b).catch((err) => {
      console.warn(`后端 ${b} 初始化失败`, err);
      return false;
    });
    if (!ok) continue;
    const notices: string[] = [];
    if (b !== order[0]) notices.push(`${BACKEND_LABEL[order[0]]} 不可用，已改用 ${BACKEND_LABEL[b]}`);
    if (b === 'cpu') notices.push('未使用 GPU 加速，训练会非常慢');
    return { backend: b, notices };
  }
  throw new Error('没有可用的计算后端');
}

// 后端包按需加载，用不到的不下载
const LOADERS: Record<BackendName, () => Promise<unknown>> = {
  webgpu: () => import('@tensorflow/tfjs-backend-webgpu'),
  webgl: () => import('@tensorflow/tfjs-backend-webgl'),
  cpu: () => import('@tensorflow/tfjs-backend-cpu'),
};

export async function initBackend(search: string): Promise<{ backend: BackendName; notices: string[] }> {
  const { order, notices } = backendOrder(search);
  const chosen = await chooseBackend(order, async (b) => {
    await LOADERS[b]();
    const ok = await tf.setBackend(b);
    await tf.ready();
    return ok;
  });
  return { backend: chosen.backend, notices: [...notices, ...chosen.notices] };
}
