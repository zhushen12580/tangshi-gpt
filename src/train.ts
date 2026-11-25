// 训练控制器：在主线程里一步一步地训练，每步之后让出主线程，界面保持响应

import * as tf from '@tensorflow/tfjs-core';
import { makeBatch } from './data';
import { initParams, lossFn, type GPTConfig, type Params } from './model';

// ---------- 训练常量 ----------

export const LEARNING_RATE = 2e-3;
export const BATCH_SIZE = 64;
export const MILESTONE_EVERY = 50;
const CURRENT_WINDOW = 10; // 「当前 loss」取最近 10 步的平均
const SPEED_WINDOW = 20; // 「步/秒」按最近 20 步计算
const EMA_BETA = 0.9;
const VAL_CHUNK = 64; // 验证集按 64 首一块做前向，显存占用不超过训练的一个 batch

export type TrainState = 'idle' | 'running' | 'paused' | 'nan';

export interface TrainStats {
  step: number;
  initialLoss: number | null;
  currentLoss: number | null;
  ema: number | null;
  stepsPerSec: number;
  elapsedMs: number;
}

export interface TrainerOptions {
  cfg: GPTConfig;
  train: number[][];
  val: number[][];
  bosId: number;
  batchSize?: number;
  lr?: number;
  seed?: number;
  rng?: () => number;
  onStep?: (s: TrainStats) => void;
  onState?: (s: TrainState) => void;
  onMilestone?: (m: { step: number; valLoss: number }) => Promise<void> | void;
}

export class Trainer {
  readonly cfg: GPTConfig;
  params: Params;
  state: TrainState = 'idle';
  lossHistory: number[] = [];
  valHistory: { step: number; loss: number }[] = [];

  private readonly o: TrainerOptions;
  private optimizer: tf.Optimizer;
  private ema: number | null = null;
  private stepTimes: number[] = []; // 最近几步结束的时刻，算步/秒
  private elapsedBefore = 0; // 之前各段训练累计的时长
  private runStartedAt: number | null = null; // 本段训练开始的时刻（不在训练中时为 null）
  private loop: Promise<void> | null = null;
  private stopRequested = false;

  constructor(o: TrainerOptions) {
    this.o = o;
    this.cfg = o.cfg;
    this.params = initParams(o.cfg, o.seed);
    this.optimizer = tf.train.adam(o.lr ?? LEARNING_RATE);
  }

  stats(): TrainStats {
    const h = this.lossHistory;
    const recent = h.slice(-CURRENT_WINDOW);
    const t = this.stepTimes;
    return {
      step: h.length,
      initialLoss: h.length ? h[0] : null,
      currentLoss: recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : null,
      ema: this.ema,
      stepsPerSec: t.length > 1 ? ((t.length - 1) * 1000) / (t[t.length - 1] - t[0]) : 0,
      elapsedMs: this.elapsedBefore + (this.runStartedAt === null ? 0 : performance.now() - this.runStartedAt),
    };
  }

  /** 训练一步：随机取一个 batch，前向、反向、Adam 更新，返回这一步的 loss */
  async step(): Promise<number> {
    const B = this.o.batchSize ?? BATCH_SIZE;
    const { inputs, targets } = makeBatch(this.o.train, this.o.bosId, B, this.o.rng);
    const T = inputs.length / B;
    const loss = tf.tidy(() => {
      const x = tf.tensor2d(inputs, [B, T], 'int32');
      const y = tf.tensor2d(targets, [B, T], 'int32');
      return this.optimizer.minimize(() => lossFn(this.params, this.cfg, x, y), true, Object.values(this.params))!;
    });
    const value = (await loss.data())[0];
    loss.dispose();

    this.lossHistory.push(value);
    this.ema = this.ema === null ? value : EMA_BETA * this.ema + (1 - EMA_BETA) * value;
    this.stepTimes = [...this.stepTimes, performance.now()].slice(-(SPEED_WINDOW + 1));
    this.o.onStep?.(this.stats());
    return value;
  }

  /** 在整个验证集上做前向（不求梯度），返回平均 loss */
  async evalValLoss(): Promise<number> {
    const { val, bosId } = this.o;
    let sum = 0;
    for (let i = 0; i < val.length; i += VAL_CHUNK) {
      const chunk = val.slice(i, i + VAL_CHUNK);
      const loss = tf.tidy(() => {
        const x = tf.tensor2d(chunk.map((p) => [bosId, ...p.slice(0, -1)]), undefined, 'int32');
        const y = tf.tensor2d(chunk, undefined, 'int32');
        return lossFn(this.params, this.cfg, x, y);
      });
      sum += (await loss.data())[0] * chunk.length; // 每首字数相同，按首数加权就是整体平均
      loss.dispose();
    }
    return sum / val.length;
  }

  /** 开始或继续训练；已经在跑时什么都不做，保证同一时刻只有一个循环 */
  start(): void {
    if (this.loop) return;
    this.stopRequested = false;
    this.runStartedAt = performance.now();
    this.stepTimes = [this.runStartedAt]; // 暂停的时间不算进步/秒
    this.setState('running');
    this.loop = this.run();
  }

  /** 请求暂停，等当前这一步结束、循环退出后才返回 */
  async pause(): Promise<void> {
    this.stopRequested = true;
    await this.loop;
  }

  dispose(): void {
    Object.values(this.params).forEach((v) => v.dispose());
    this.optimizer.dispose();
  }

  private async run(): Promise<void> {
    try {
      await this.milestoneIfDue(); // 第 0 步的里程碑在第一步训练之前
      while (!this.stopRequested) {
        await this.step(); // 通过 this.step 调用，单测会 spy 它
        await this.milestoneIfDue(); // 紧跟在这一步后面：停在第 50 步时，它的里程碑也已经做完
        await new Promise((r) => setTimeout(r, 0)); // 让出主线程，界面才能刷新、响应点击
      }
    } finally {
      this.elapsedBefore += performance.now() - this.runStartedAt!;
      this.runStartedAt = null;
      this.loop = null;
      this.setState('paused');
    }
  }

  /** 步数是 50 的倍数且这一步还没做过里程碑时：算验证 loss，记下来，再等回调（采样等）做完 */
  private async milestoneIfDue(): Promise<void> {
    const step = this.lossHistory.length;
    if (step % MILESTONE_EVERY !== 0 || this.valHistory.at(-1)?.step === step) return;
    const valLoss = await this.evalValLoss();
    this.valHistory.push({ step, loss: valLoss });
    await this.o.onMilestone?.({ step, valLoss });
  }

  private setState(s: TrainState): void {
    this.state = s;
    this.o.onState?.(s);
  }
}
