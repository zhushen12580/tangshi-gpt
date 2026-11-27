import { expect, test, vi } from 'vitest';
import {
  deleteCheckpoint, isCompatible, loadCheckpoint, makeCheckpoint, saveCheckpoint,
  StorageUnavailableError, storageErrorMessage,
} from '../../src/checkpoint';

const config = { vocabSize: 11, ctx: 8, nLayer: 2, nHead: 2, dModel: 16, dFF: 32 };
const snapshot = {
  step: 0, weights: { wte: new Float32Array([0.1]) }, optimizer: [],
  lossHistory: [], valHistory: [], elapsedMs: 0,
};
const checkpoint = () => makeCheckpoint(snapshot, '01234567', config, [], 123);

test('存档保留二进制权重，只兼容同版本、字表和模型配置', () => {
  const c = checkpoint();
  expect(c).toEqual({ ...snapshot, version: 1, config, vocabHash: '01234567', samples: [], savedAt: 123 });
  expect(isCompatible(c, c.vocabHash, { ...config })).toBe(true);
  expect(isCompatible(c, '76543210', config)).toBe(false);
  for (const key of Object.keys(config) as (keyof typeof config)[]) {
    expect(isCompatible(c, c.vocabHash, { ...config, [key]: config[key] + 1 })).toBe(false);
  }
  const futureVersion = structuredClone(c);
  Object.assign(futureVersion, { version: 2 });
  expect(isCompatible(futureVersion, c.vocabHash, config)).toBe(false);
});

test('存档复制生成记录，之后改动原记录不影响存档', () => {
  const sample = {
    step: 0, valLoss: null, temperature: 0.8, constrained: true, manual: false,
    text: '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。', formatOk: true,
  };
  const samples = [sample];
  const c = makeCheckpoint(snapshot, '01234567', config, samples, 123);
  sample.text = 'x';
  samples.push({ ...sample });
  expect(c.samples).toHaveLength(1);
  expect(c.samples[0]).not.toBe(sample);
  expect(c.samples[0].text).toBe('白日依山尽，黄河入海流。欲穷千里目，更上一层楼。');
});

test('保存失败时区分存储不可用、配额不足和其他错误', () => {
  expect(storageErrorMessage(new DOMException('x', 'QuotaExceededError'))).toBe('无法保存：浏览器存储空间不足');
  expect(storageErrorMessage(new StorageUnavailableError())).toBe('无法保存：此浏览器不支持本地存储');
  expect(storageErrorMessage(new Error('x'))).toBe('无法保存：x');
});

/** 够 checkpoint.ts 用的内存版 IndexedDB：回调都异步触发，和浏览器一样 */
function fakeIndexedDB(data: Map<string, unknown>): IDBFactory {
  const open = () => {
    const request = {} as IDBOpenDBRequest & { result: IDBDatabase };
    const transaction = () => {
      const tx = {} as IDBTransaction;
      let pending = 0;
      // 请求先触发 onsuccess（回调里还能在同一个事务里发新请求），全部做完才触发 oncomplete
      const run = (fn: () => unknown) => {
        const request = { result: fn() } as IDBRequest;
        pending += 1;
        setTimeout(() => {
          request.onsuccess?.(new Event('success'));
          if (--pending === 0) tx.oncomplete?.(new Event('complete'));
        });
        return request;
      };
      tx.objectStore = () => ({
        get: (k: string) => run(() => data.get(k)),
        put: (v: unknown, k: string) => run(() => data.set(k, v) && k),
        delete: (k: string) => run(() => void data.delete(k)),
      }) as unknown as IDBObjectStore;
      return tx;
    };
    setTimeout(() => {
      Object.assign(request, { result: { transaction, close: () => {} } });
      request.onsuccess?.(new Event('success'));
    });
    return request;
  };
  return { open } as unknown as IDBFactory;
}

function fakeLocalStorage(): Storage {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
  } as Storage;
}

test('确认删除时先同步记下，删除没来得及提交就离开页面，下次读到同一个存档也当已删除', async () => {
  const ls = fakeLocalStorage();
  vi.stubGlobal('localStorage', ls);
  try {
    const data = new Map<string, unknown>();
    const idb = fakeIndexedDB(data);
    await saveCheckpoint(checkpoint(), idb);
    // 删除请求要等 IndexedDB 异步回调才发出；记号必须在调用的这一刻就写好（verify 发现 F-3951d65ec84e）
    const pending = deleteCheckpoint(123, fakeIndexedDB(new Map())); // 模拟“这次删除丢了”：真正的数据没动
    expect(ls.getItem('tangshi-gpt:deleted')).toBe('123');
    await pending;
    ls.setItem('tangshi-gpt:deleted', '123'); // 页面在删除完成前就走了，记号还在
    expect(data.has('latest')).toBe(true);
    await expect(loadCheckpoint(idb)).resolves.toBeNull();
    expect(data.has('latest')).toBe(false); // 补删了
    expect(ls.getItem('tangshi-gpt:deleted')).toBeNull();

    // 之后新存的档（savedAt 不同）不受旧记号影响
    ls.setItem('tangshi-gpt:deleted', '123');
    await saveCheckpoint(makeCheckpoint(snapshot, '01234567', config, [], 456), idb);
    await expect(loadCheckpoint(idb)).resolves.toMatchObject({ savedAt: 456 });

    // 正常删除：删完记号清掉
    await deleteCheckpoint(456, idb);
    expect(data.has('latest')).toBe(false);
    expect(ls.getItem('tangshi-gpt:deleted')).toBeNull();
  } finally {
    vi.unstubAllGlobals();
  }
});

test('补删前另一个标签页刚存了新档，新档留着（release 发现 F-79cf48a1ec23）', async () => {
  const ls = fakeLocalStorage();
  vi.stubGlobal('localStorage', ls);
  try {
    const data = new Map<string, unknown>();
    const idb = fakeIndexedDB(data);
    await saveCheckpoint(checkpoint(), idb);
    ls.setItem('tangshi-gpt:deleted', '123'); // 上次确认删除第 123 号，没来得及删完
    const newer = makeCheckpoint(snapshot, '01234567', config, [], 456);
    let opens = 0;
    const racing = {
      open: (name: string, version?: number) => {
        // 读到旧存档之后、补删的写事务开始之前，另一个标签页提交了新存档
        if (++opens === 2) data.set('latest', newer);
        return idb.open(name, version);
      },
    } as IDBFactory;
    await expect(loadCheckpoint(racing)).resolves.toBeNull();
    expect(data.get('latest')).toBe(newer);
    expect(ls.getItem('tangshi-gpt:deleted')).toBeNull();
    await expect(loadCheckpoint(idb)).resolves.toBe(newer);

    // 用户确认删除第 123 号时，存储里已经是别的标签页的新档：同样不删，并告诉调用方没删
    await expect(deleteCheckpoint(123, idb)).resolves.toBe(false);
    expect(data.get('latest')).toBe(newer);
    await expect(deleteCheckpoint(456, idb)).resolves.toBe(true);
    expect(data.has('latest')).toBe(false);
  } finally {
    vi.unstubAllGlobals();
  }
});

test('IndexedDB 不可用时拒绝保存，读取和删除按无存档处理', async () => {
  vi.stubGlobal('indexedDB', undefined);
  try {
    await expect(saveCheckpoint(checkpoint(), undefined)).rejects.toBeInstanceOf(StorageUnavailableError);
    await expect(loadCheckpoint(undefined)).resolves.toBeNull();
    await expect(deleteCheckpoint(undefined)).resolves.toBe(true);
  } finally {
    vi.unstubAllGlobals();
  }
});
