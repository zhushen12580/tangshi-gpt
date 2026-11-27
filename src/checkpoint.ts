import type { GPTConfig } from './model';
import type { SampleRecord } from './sample';
import type { TrainerSnapshot } from './train';

export const CHECKPOINT_VERSION = 1;
export const DB_NAME = 'tangshi-gpt';
export const STORE = 'checkpoints';
export const KEY = 'latest';

export interface Checkpoint extends TrainerSnapshot {
  version: 1;
  vocabHash: string;
  config: GPTConfig;
  samples: SampleRecord[];
  savedAt: number;
}

export function makeCheckpoint(
  s: TrainerSnapshot, vocabHash: string, config: GPTConfig, samples: SampleRecord[], now = Date.now(),
): Checkpoint {
  return {
    ...s, version: CHECKPOINT_VERSION, vocabHash, config: { ...config },
    samples: samples.map((sample) => ({ ...sample })), savedAt: now,
  };
}

export function isCompatible(c: Checkpoint, vocabHash: string, config: GPTConfig): boolean {
  return c.version === CHECKPOINT_VERSION && c.vocabHash === vocabHash && !!c.config
    && (['vocabSize', 'ctx', 'nLayer', 'nHead', 'dModel', 'dFF'] as const)
      .every((key) => c.config[key] === config[key]);
}

export class StorageUnavailableError extends Error {}

export function storageErrorMessage(e: unknown): string {
  if (e instanceof StorageUnavailableError) return '无法保存：此浏览器不支持本地存储';
  if (e && typeof e === 'object' && 'name' in e && e.name === 'QuotaExceededError') {
    return '无法保存：浏览器存储空间不足';
  }
  const message = e && typeof e === 'object' && 'message' in e ? e.message : e;
  return `无法保存：${String(message)}`;
}

async function transact<T>(
  idb: IDBFactory, mode: IDBTransactionMode, operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = idb.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const request = operation(tx.objectStore(STORE));
      // 请求成功后事务仍可能失败，只在提交后报告成功。
      tx.oncomplete = () => resolve(request.result);
      tx.onabort = () => reject(tx.error ?? request.error ?? new Error('存档事务已中止'));
    });
  } finally {
    db.close();
  }
}

export async function saveCheckpoint(c: Checkpoint, idb = globalThis.indexedDB): Promise<void> {
  if (!idb) throw new StorageUnavailableError();
  await transact(idb, 'readwrite', (store) => store.put(c, KEY));
}

const DELETED_KEY = 'tangshi-gpt:deleted'; // localStorage：已确认删除、可能还没删掉的存档的 savedAt
// localStorage 被禁用或不存在时访问就抛错，这时不记号，只做原来的异步删除
const withStorage = <T>(fn: (ls: Storage) => T): T | undefined => {
  try {
    return fn(globalThis.localStorage);
  } catch {
    return undefined;
  }
};

export async function loadCheckpoint(idb = globalThis.indexedDB): Promise<Checkpoint | null> {
  if (!idb) return null;
  const c = (await transact<Checkpoint | undefined>(idb, 'readonly', (store) => store.get(KEY))) ?? null;
  if (c && withStorage((ls) => ls.getItem(DELETED_KEY)) === String(c.savedAt)) {
    await deleteCheckpoint(c.savedAt, idb); // 上次确认删除后没来得及删完：补删，当作没有存档
    return null;
  }
  return c;
}

/**
 * 删除要等 IndexedDB 的异步回调才发出请求。确认后马上刷新、主线程又正忙时，新页面先提交，
 * 删除请求根本没发出去，存档又回来了（verify 发现 F-3951d65ec84e）。所以调用时先同步记下 savedAt，
 * 删完再清掉；loadCheckpoint 读到的还是这个存档，就补删。不给 savedAt（不兼容、没有 savedAt 的存档）就直接删
 */
export async function deleteCheckpoint(savedAt?: number, idb = globalThis.indexedDB): Promise<void> {
  if (savedAt !== undefined) withStorage((ls) => ls.setItem(DELETED_KEY, String(savedAt)));
  // 读和删在同一个写事务里：只删 savedAt 还对得上的存档，别的标签页刚存的新档留着（release 发现 F-79cf48a1ec23）
  if (idb) await transact(idb, 'readwrite', (store): IDBRequest => {
    if (savedAt === undefined) return store.delete(KEY);
    const request = store.get(KEY);
    request.onsuccess = () => {
      if ((request.result as Checkpoint | undefined)?.savedAt === savedAt) store.delete(KEY);
    };
    return request;
  });
  withStorage((ls) => ls.removeItem(DELETED_KEY));
}
