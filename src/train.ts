// 训练控制器：在主线程里一步一步地训练，每步之后让出主线程，界面保持响应

import * as tf from '@tensorflow/tfjs-core';
import { makeBatch } from './data';
import { initParams, lossFn, type GPTConfig, type Params } from './model';

// ---------- 训练常量 ----------

export const LEARNING_RATE = 2e-3;
export const BATCH_SIZE = 64;
export const CPU_BATCH_SIZE = 32; // CPU 后端 batch 64 一步约 30 秒，减半后第 1 步才能稳定在点击后 60 秒内出来
export const CPU_MICRO_BATCH = 1; // CPU 后端每次只算 1 行（约 0.5 秒）就让出主线程，「暂停」才能在 2 秒内响应
export const MILESTONE_EVERY = 50;
const CURRENT_WINDOW = 10; // 「当前 loss」取最近 10 步的平均
const SPEED_WINDOW = 20; // 「步/秒」按最近 20 步计算
const EMA_BETA = 0.9;
const VAL_CHUNK = 64; // 验证集按 64 首一块做前向，显存占用不超过训练的一个 batch

/** 让出主线程：排到点击、重绘等任务之后再接着算 */
const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

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
  /** 设了就把每个 batch 切成这么多行一块分别求梯度，块与块之间让出主线程（CPU 后端用）；验证集也按这个大小分块 */
  microBatch?: number;
  lr?: number;
  seed?: number;
  rng?: () => number;
  onStep?: (s: TrainStats) => void;
  onState?: (s: TrainState) => void;
  /** 里程碑一开始、算验证 loss 之前调用（写诗用）：软件 WebGL 下验证集要算两三秒，第 0 步的诗点了「开始训练」就要出来 */
  onMilestoneStart?: (step: number) => Promise<void> | void;
  onMilestone?: (m: { step: number; valLoss: number }) => Promise<void> | void;
}

export interface TrainerSnapshot {
  step: number;
  weights: Record<string, Float32Array>;
  optimizer: { name: string; shape: number[]; data: Float32Array }[];
  lossHistory: number[];
  valHistory: { step: number; loss: number }[];
  elapsedMs: number;
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
  private lastMilestoneStep: number | null = null;
  private stepTimes: number[] = []; // 最近几步结束的时刻，算步/秒
  private elapsedBefore = 0; // 之前各段训练累计的时长
  private runStartedAt: number | null = null; // 本段训练开始的时刻（不在训练中时为 null）
  private loop: Promise<void> | null = null;
  private stopRequested = false;
  private chain: Promise<unknown> = Promise.resolve(); // 训练的每一步和插队任务都排在这条链上，彼此串行

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

  /**
   * 训练一步：随机取一个 batch，前向、反向、Adam 更新，返回这一步的 loss。
   * 分块训练时，这一步算到一半收到暂停请求就整步放弃（权重不动），返回 null
   */
  async step(): Promise<number | null> {
    const B = this.o.batchSize ?? BATCH_SIZE;
    const { inputs, targets } = makeBatch(this.o.train, this.o.bosId, B, this.o.rng);
    const value = this.o.microBatch ? await this.microBatchUpdate(inputs, targets, B) : await this.batchUpdate(inputs, targets, B);
    if (value === null || !Number.isFinite(value)) return value; // 数值异常：不记进历史，由 run() 停下循环

    this.lossHistory.push(value);
    this.ema = this.ema === null ? value : EMA_BETA * this.ema + (1 - EMA_BETA) * value;
    this.stepTimes = [...this.stepTimes, performance.now()].slice(-(SPEED_WINDOW + 1));
    this.o.onStep?.(this.stats());
    return value;
  }

  /** 整个 batch 一次前向、反向、更新 */
  private async batchUpdate(inputs: Int32Array, targets: Int32Array, B: number): Promise<number> {
    const T = inputs.length / B;
    const loss = tf.tidy(() => {
      const x = tf.tensor2d(inputs, [B, T], 'int32');
      const y = tf.tensor2d(targets, [B, T], 'int32');
      return this.optimizer.minimize(() => lossFn(this.params, this.cfg, x, y), true, Object.values(this.params))!;
    });
    const value = (await loss.data())[0];
    loss.dispose();
    return value;
  }

  /**
   * CPU 后端一整个 batch 要连续算十几秒，期间点击没有反应。这里切成 microBatch 行一块，
   * 逐块求梯度累加，最后用总梯度更新一次（结果和整个 batch 一样）；块与块之间让出主线程并检查暂停
   */
  private async microBatchUpdate(inputs: Int32Array, targets: Int32Array, B: number): Promise<number | null> {
    const T = inputs.length / B;
    const M = this.o.microBatch!;
    const vars = Object.values(this.params);
    let sum: tf.NamedTensorMap = {};
    let value = 0;
    try {
      for (let r = 0; r < B; r += M) {
        const n = Math.min(M, B - r);
        const { value: loss, grads } = tf.tidy(() => {
          const rows = (a: Int32Array) => tf.tensor2d(a.subarray(r * T, (r + n) * T), [n, T], 'int32');
          const x = rows(inputs);
          const y = rows(targets);
          // 按行数加权：各块的 loss 和梯度加起来，正好是整个 batch 的平均
          return tf.variableGrads(() => tf.mul(lossFn(this.params, this.cfg, x, y), n / B), vars);
        });
        value += (await loss.data())[0];
        const prev = sum;
        sum = tf.tidy(() => Object.fromEntries(Object.entries(grads).map(([k, g]) => [k, prev[k] ? tf.add(prev[k], g) : tf.clone(g)])));
        tf.dispose([loss, ...Object.values(prev), ...Object.values(grads)]);
        await yieldToUI();
        if (this.stopRequested) return null;
      }
      this.optimizer.applyGradients(sum); // Adam 更新也要算一会儿，单独占一段
      return value;
    } finally {
      tf.dispose(Object.values(sum));
    }
  }

  /** 在整个验证集上做前向（不求梯度），返回平均 loss */
  async evalValLoss(): Promise<number> {
    const { val, bosId } = this.o;
    const size = this.o.microBatch ?? VAL_CHUNK;
    let sum = 0;
    for (let i = 0; i < val.length; i += size) {
      if (i > 0) await yieldToUI();
      const chunk = val.slice(i, i + size);
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

  /**
   * 在两步之间执行 fn（立即生成、保存、查看注意力用）：训练中排到当前这一步（含里程碑）之后、下一步之前；
   * 不在训练时立即执行。各任务之间也是串行的，不会并发读写权重
   */
  runBetweenSteps<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.chain.then(fn);
    this.chain = result.catch(() => {}); // 出错只影响调用方，不卡住后面的任务
    return result;
  }

  /**
   * 开始或继续训练；已经在跑时什么都不做，保证同一时刻只有一个循环。
   * 暂停请求还没生效（循环还在做这一步的里程碑）时，撤销请求，接着用原来的循环
   */
  start(): void {
    if (this.loop && !this.stopRequested) return;
    this.stopRequested = false;
    this.runStartedAt = performance.now();
    this.stepTimes = [this.runStartedAt]; // 暂停的时间不算进步/秒
    this.setState('running');
    this.loop ??= this.run();
  }

  /**
   * 请求暂停：状态和计时立刻停下（里程碑在软件 WebGL 下要做十几秒，按钮不能等它），
   * 返回的 Promise 等当前这一步（含里程碑）结束、循环退出后才完成
   */
  async pause(): Promise<void> {
    if (this.loop && !this.stopRequested) {
      this.stopRequested = true;
      this.stopClock();
      this.setState('paused');
    }
    await this.loop;
  }

  /** 训练中通过 runBetweenSteps 调用，确保所有数据来自同一步。 */
  async snapshot(): Promise<TrainerSnapshot> {
    const { step, elapsedMs } = this.stats();
    const weights: TrainerSnapshot['weights'] = {};
    for (const [name, tensor] of Object.entries(this.params)) {
      weights[name] = new Float32Array(await tensor.data());
    }
    const adamWeights = await this.optimizer.getWeights();
    try {
      const optimizer: TrainerSnapshot['optimizer'] = [];
      for (const { name, tensor } of adamWeights) {
        optimizer.push({ name, shape: [...tensor.shape], data: new Float32Array(await tensor.data()) });
      }
      return {
        step, weights, optimizer, elapsedMs,
        lossHistory: [...this.lossHistory],
        valHistory: this.valHistory.map((point) => ({ ...point })),
      };
    } finally {
      // Adam 只为迭代次数创建临时张量，其余是优化器持有的矩变量。
      adamWeights[0].tensor.dispose();
    }
  }

  async restore(s: TrainerSnapshot): Promise<void> {
    await this.pause();
    await this.runBetweenSteps(async () => {
      const entries = Object.entries(this.params);
      const { weights, adamWeights } = tf.tidy(() => ({
        weights: entries.map(([name, variable]) => tf.tensor(s.weights[name].slice(), variable.shape)),
        adamWeights: s.optimizer.map(({ name, data, shape }, i) => ({
          name, tensor: tf.tensor(data.slice(), shape, i === 0 ? 'int32' : 'float32'),
        })),
      }));
      // setWeights 会替换矩变量，但不会释放旧变量。
      const optimizer = tf.train.adam(this.o.lr ?? LEARNING_RATE);
      try {
        await optimizer.setWeights(adamWeights);
        entries.forEach(([, variable], i) => variable.assign(weights[i]));
      } catch (error) {
        optimizer.dispose();
        throw error;
      } finally {
        tf.dispose(weights);
        tf.dispose(adamWeights.map(({ tensor }) => tensor));
      }
      this.optimizer.dispose();
      this.optimizer = optimizer;
      this.lossHistory = [...s.lossHistory];
      this.valHistory = s.valHistory.map((point) => ({ ...point }));
      this.elapsedBefore = s.elapsedMs;
      this.stepTimes = [];
      this.ema = this.lossHistory.reduce<number | null>(
        (ema, loss) => ema === null ? loss : EMA_BETA * ema + (1 - EMA_BETA) * loss, null,
      );
      // 最后一个验证点就是做过的最后一个里程碑；训练前保存的存档没有验证点，继续时要补做第 0 步
      this.lastMilestoneStep = s.valHistory.at(-1)?.step ?? null;
      this.setState('paused');
    });
  }

  /** 回到第 0 步：先暂停，再重新初始化权重和 Adam、清空历史；不碰存档 */
  async reset(): Promise<void> {
    await this.pause();
    await this.runBetweenSteps(async () => {
      this.dispose();
      this.params = initParams(this.cfg, this.o.seed);
      this.optimizer = tf.train.adam(this.o.lr ?? LEARNING_RATE);
      this.lossHistory = [];
      this.valHistory = [];
      this.ema = null;
      this.lastMilestoneStep = null;
      this.stepTimes = [];
      this.elapsedBefore = 0;
      this.setState('idle');
    });
  }

  dispose(): void {
    Object.values(this.params).forEach((v) => v.dispose());
    this.optimizer.dispose();
  }

  private async run(): Promise<void> {
    let nan = false;
    try {
      // 每一步也排在 runBetweenSteps 的链上，插队的任务就只会落在两步之间
      await this.runBetweenSteps(() => this.milestoneIfDue()); // 第 0 步的里程碑在第一步训练之前
      while (!this.stopRequested && !nan) {
        await this.runBetweenSteps(async () => {
          if (this.stopRequested) return; // 等待采样时也可能收到暂停请求
          const loss = await this.step(); // 通过 this.step 调用，单测会 spy 它
          if (loss === null) return; // 算到一半暂停了，这一步已放弃
          nan = !Number.isFinite(loss);
          if (!nan) await this.milestoneIfDue(); // 紧跟在这一步后面：停在第 50 步时，它的里程碑也已经做完
        });
        await yieldToUI(); // 界面才能刷新、响应点击
      }
    } finally {
      this.stopClock();
      this.loop = null;
      this.setState(nan ? 'nan' : 'paused');
    }
  }

  /** 把这一段训练的时长记进累计，停止计时 */
  private stopClock(): void {
    if (this.runStartedAt === null) return;
    this.elapsedBefore += performance.now() - this.runStartedAt;
    this.runStartedAt = null;
  }

  /** 步数是 50 的倍数且这一步还没做过里程碑时：先等 onMilestoneStart（写诗），再算验证 loss、记下来，最后等 onMilestone 做完 */
  private async milestoneIfDue(): Promise<void> {
    const step = this.lossHistory.length;
    if (step % MILESTONE_EVERY !== 0 || this.lastMilestoneStep === step) return;
    await this.o.onMilestoneStart?.(step);
    const valLoss = await this.evalValLoss();
    this.valHistory.push({ step, loss: valLoss });
    this.lastMilestoneStep = step;
    await this.o.onMilestone?.({ step, valLoss });
  }

  private setState(s: TrainState): void {
    if (this.state === s) return; // 请求暂停和循环退出都会设 paused，只通知一次
    this.state = s;
    this.o.onState?.(s);
  }
}
