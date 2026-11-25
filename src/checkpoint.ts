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

export async function loadCheckpoint(idb = globalThis.indexedDB): Promise<Checkpoint | null> {
  if (!idb) return null;
  return (await transact<Checkpoint | undefined>(idb, 'readonly', (store) => store.get(KEY))) ?? null;
}

export async function deleteCheckpoint(idb = globalThis.indexedDB): Promise<void> {
  if (idb) await transact(idb, 'readwrite', (store) => store.delete(KEY));
}
