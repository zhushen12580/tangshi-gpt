// 一个文件读懂 GPT：只用 @tensorflow/tfjs-core 的基础张量算子手写的字符级 GPT。
// 结构：词嵌入 + 位置嵌入 → N 个 Transformer 块（pre-LN）→ 最后的 LayerNorm → 与词嵌入共享权重的输出层。

import * as tf from '@tensorflow/tfjs-core';
// 只用 tfjs-core 时梯度函数不会自动注册，反向传播（optimizer.minimize）必须先导入这一行
import '@tensorflow/tfjs-core/dist/register_all_gradients';

// ---------- 模型常量 ----------

export interface GPTConfig {
  vocabSize: number; // 字表大小 V
  ctx: number; // 上下文长度（最多看多少个字）
  nLayer: number; // Transformer 块的层数
  nHead: number; // 每层注意力头数
  dModel: number; // 隐藏维度 d
  dFF: number; // 前馈网络中间维度
}

/** 默认配置：3 层、4 头、d=128、FFN 512、上下文 24（<s> + 23 个字预测 24 个字） */
export const MODEL_DEFAULTS = { ctx: 24, nLayer: 3, nHead: 4, dModel: 128, dFF: 512 };

const INIT_STD = 0.02; // 权重初始化的标准差
const LN_EPS = 1e-5; // LayerNorm 防止除零
const MASK_NEG = -1e9; // 因果遮罩：看不到的位置加一个极小的数，softmax 后约等于 0

/** 参数表：键是自己起的名字（wte、h0.attn.wqkv …），变量本身不命名，方便重置后重建 */
export type Params = Record<string, tf.Variable>;

// ---------- 初始化 ----------

/**
 * 按固定顺序创建全部参数。
 * 权重 ~ N(0, 0.02)；残差分支的输出投影（wo、w2）缩小为 0.02/√(2·nLayer)，让深层残差累加后方差不爆；
 * 偏置为 0；LayerNorm 的 g=1、b=0。给了 seed 时，第 k 个参数用 seed + k，结果可复现。
 */
export function initParams(cfg: GPTConfig, seed?: number): Params {
  const { vocabSize: V, ctx, nLayer, dModel: d, dFF } = cfg;
  const residStd = INIT_STD / Math.sqrt(2 * nLayer);
  // [键, 形状, 初始化方式]：数字表示正态分布的标准差，'zeros'/'ones' 表示常数
  const specs: [string, number[], number | 'zeros' | 'ones'][] = [
    ['wte', [V, d], INIT_STD],
    ['wpe', [ctx, d], INIT_STD],
  ];
  for (let i = 0; i < nLayer; i++) {
    const h = `h${i}`;
    specs.push(
      [`${h}.ln1.g`, [d], 'ones'],
      [`${h}.ln1.b`, [d], 'zeros'],
      [`${h}.attn.wqkv`, [d, 3 * d], INIT_STD],
      [`${h}.attn.bqkv`, [3 * d], 'zeros'],
      [`${h}.attn.wo`, [d, d], residStd],
      [`${h}.attn.bo`, [d], 'zeros'],
      [`${h}.ln2.g`, [d], 'ones'],
      [`${h}.ln2.b`, [d], 'zeros'],
      [`${h}.mlp.w1`, [d, dFF], INIT_STD],
      [`${h}.mlp.b1`, [dFF], 'zeros'],
      [`${h}.mlp.w2`, [dFF, d], residStd],
      [`${h}.mlp.b2`, [d], 'zeros'],
    );
  }
  specs.push(['lnf.g', [d], 'ones'], ['lnf.b', [d], 'zeros']);

  // tidy 会回收初始值张量；变量本身不会被 tidy 回收
  return tf.tidy(() =>
    Object.fromEntries(
      specs.map(([key, shape, init], k) => {
        const value =
          init === 'zeros'
            ? tf.zeros(shape)
            : init === 'ones'
              ? tf.ones(shape)
              : tf.randomNormal(shape, 0, init, 'float32', seed === undefined ? undefined : seed + k);
        return [key, tf.variable(value, true)];
      }),
    ),
  );
}

export function countParams(p: Params): number {
  return Object.values(p).reduce((n, v) => n + v.size, 0);
}

// ---------- 基础层 ----------

/** 全连接：x [N,in] · w [in,out] + b [out] */
function dense(x: tf.Tensor2D, w: tf.Tensor, b: tf.Tensor): tf.Tensor2D {
  return tf.add(tf.matMul(x, w as tf.Tensor2D), b);
}

/** LayerNorm：沿最后一维减均值、除标准差，再乘 g 加 b */
function layerNorm(x: tf.Tensor2D, g: tf.Tensor, b: tf.Tensor): tf.Tensor2D {
  const { mean, variance } = tf.moments(x, -1, true);
  return tf.add(tf.mul(tf.mul(tf.sub(x, mean), tf.rsqrt(tf.add(variance, LN_EPS))), g), b);
}

/** GELU 的 tanh 近似：0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))) */
function gelu(x: tf.Tensor2D): tf.Tensor2D {
  const inner = tf.mul(Math.sqrt(2 / Math.PI), tf.add(x, tf.mul(0.044715, tf.pow(x, 3))));
  return tf.mul(tf.mul(0.5, x), tf.add(1, tf.tanh(inner)));
}

// ---------- 注意力 ----------

/**
 * 多头因果自注意力。x 是展平的 [B·T, d]。
 * 返回输出 [B·T, d] 和注意力概率 [B,H,T,T]（热力图要用）。
 */
function attention(
  p: Params,
  i: number,
  x: tf.Tensor2D,
  B: number,
  T: number,
  cfg: GPTConfig,
): { out: tf.Tensor2D; probs: tf.Tensor4D } {
  const { nHead: H, dModel: d } = cfg;
  const hd = d / H; // 每个头的维度
  const h = `h${i}.attn`;

  // 一次矩阵乘法同时算出 q、k、v，再拆成 3 份
  const qkv = dense(x, p[`${h}.wqkv`], p[`${h}.bqkv`]);
  // [B·T, d] → [B, T, H, hd] → [B, H, T, hd]：每个头各自独立做注意力
  const toHeads = (t: tf.Tensor) => tf.transpose(tf.reshape(t, [B, T, H, hd]), [0, 2, 1, 3]) as tf.Tensor4D;
  const [q, k, v] = tf.split(qkv, 3, -1).map(toHeads);

  // 打分：q·kᵀ / √hd，形状 [B,H,T,T]，第 t 行表示第 t 个字看各个字的程度
  const scores = tf.div(tf.matMul(q, k, false, true), Math.sqrt(hd));
  // 因果遮罩：下三角为 1（能看到自己和前面的字），右上三角加 −1e9（看不到后面的字）
  const allowed = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
  const masked = tf.add(scores, tf.mul(tf.sub(1, allowed), MASK_NEG));
  const probs = tf.softmax(masked) as tf.Tensor4D; // softmax 沿最后一维，每行和为 1

  // 按概率加权求和 v，再把各头拼回 [B·T, d]，经过输出投影
  const y = tf.reshape(tf.transpose(tf.matMul(probs, v), [0, 2, 1, 3]), [B * T, d]) as tf.Tensor2D;
  return { out: dense(y, p[`${h}.wo`], p[`${h}.bo`]), probs };
}

// ---------- 前向 ----------

/**
 * 前向传播。ids [B,T]（int32，T ≤ ctx）。
 * 返回 logits [B,T,V]（每个位置对下一个字的打分）和每层的注意力概率 [B,H,T,T]。
 */
export function forward(
  p: Params,
  cfg: GPTConfig,
  ids: tf.Tensor2D,
): { logits: tf.Tensor3D; attn: tf.Tensor4D[] } {
  return tf.tidy(() => {
    const [B, T] = ids.shape;
    const d = cfg.dModel;

    // 词嵌入 [B,T,d] + 位置嵌入 [T,d]（广播到每个样本），再展平成 [B·T, d]
    // 查表写成 oneHot · wte 而不是 gather：结果相同，但 gather 的梯度在 CPU 后端上要按字表逐行循环，
    // V≈3600 时一步要好几秒；矩阵乘法在 WebGL 上一样快
    const oneHot = tf.oneHot(tf.reshape(ids, [-1]) as tf.Tensor1D, cfg.vocabSize, 1, 0, 'float32');
    const tok = tf.reshape(tf.matMul(oneHot, p.wte), [B, T, d]);
    const pos = tf.slice(p.wpe, [0, 0], [T, d]);
    let x = tf.reshape(tf.add(tok, pos), [B * T, d]) as tf.Tensor2D;

    const attn: tf.Tensor4D[] = [];
    for (let i = 0; i < cfg.nLayer; i++) {
      const h = `h${i}`;
      // pre-LN：先归一化再进子层，子层输出加回残差
      const a = attention(p, i, layerNorm(x, p[`${h}.ln1.g`], p[`${h}.ln1.b`]), B, T, cfg);
      x = tf.add(x, a.out);
      attn.push(a.probs);
      // 前馈网络：d → dFF → GELU → d
      const m = gelu(dense(layerNorm(x, p[`${h}.ln2.g`], p[`${h}.ln2.b`]), p[`${h}.mlp.w1`], p[`${h}.mlp.b1`]));
      x = tf.add(x, dense(m, p[`${h}.mlp.w2`], p[`${h}.mlp.b2`]));
    }

    // 最后一层 LayerNorm，再与词嵌入共享权重：logits = h · wteᵀ
    const hf = layerNorm(x, p['lnf.g'], p['lnf.b']);
    const logits = tf.reshape(tf.matMul(hf, p.wte, false, true), [B, T, cfg.vocabSize]) as tf.Tensor3D;
    return { logits, attn };
  });
}

/** 平均交叉熵（单位 nats）：−mean(Σ oneHot(目标) × logSoftmax(logits)) */
export function lossFn(p: Params, cfg: GPTConfig, inputs: tf.Tensor2D, targets: tf.Tensor2D): tf.Scalar {
  return tf.tidy(() => {
    const V = cfg.vocabSize;
    const logp = tf.logSoftmax(tf.reshape(forward(p, cfg, inputs).logits, [-1, V]));
    const target = tf.oneHot(tf.reshape(targets, [-1]) as tf.Tensor1D, V);
    return tf.neg(tf.mean(tf.sum(tf.mul(target, logp), -1))) as tf.Scalar;
  });
}

/** 生成用：给定已有的 id 序列（长度 ≤ ctx），返回最后一个位置对下一个字的 logits（长度 V） */
export async function nextTokenLogits(p: Params, cfg: GPTConfig, ids: number[]): Promise<Float32Array> {
  const last = tf.tidy(() => {
    const { logits } = forward(p, cfg, tf.tensor2d([ids], [1, ids.length], 'int32'));
    return tf.reshape(tf.slice(logits, [0, ids.length - 1, 0], [1, 1, cfg.vocabSize]), [cfg.vocabSize]);
  });
  const out = (await last.data()) as Float32Array;
  last.dispose();
  return out;
}
