# 从 FlashAttention 到 FA3，再到 FlashDecoding

> 一条主线：**FA1 先解决 Attention 中间矩阵太大、搬运太多；FA2 再解决 GPU 任务分得不够好；FA3 进一步解决 Hopper 上搬数据、矩阵乘和 softmax 没有同时工作；FlashDecoding 则专门解决 decode 时 query 只有一个、GPU 没活干的问题。**

本文以两篇中文学习笔记为线索，并用论文/作者文章校对术语：

- [图解：从 Online-Softmax 到 FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)
- [原理与图解：FlashDecoding / FlashDecoding++](https://zhuanlan.zhihu.com/p/696075602)

非常感谢 DefTruth 大佬在知乎写的很多优质的博客！！
---

## 0. 先建立共同语言

单个 attention head 的计算是：

```text
S = (Q @ K.T) / sqrt(d)
P = softmax(S)          # 对每一行、沿 key 维度归一化
O = P @ V
```

其中：

| 张量 | 形状 | 含义 |
|---|---:|---|
| `Q, K, V` | `N × d` | 每个 token 一个长度为 `d` 的向量 |
| `S, P` | `N × N` | 每个 query 对每个 key 的分数/概率 |
| `O` | `N × d` | attention 输出 |

`S` 和 `P` 是 `N × N` 大小。普通实现会把它们写到 HBM（显存），之后又读回来；长上下文时，这些读写很贵。

还要区分两个推理阶段：

```text
prefill：一次处理 prompt 的很多 token，Q 的长度通常很长
decode ：每轮只生成 1 个 token，Q 的长度通常为 1，但 KV cache 很长
```

这一区别决定了 FlashAttention 和 FlashDecoding 的适用场景不同。

---

## 1. FlashAttention / FA1：不物化完整的 S 和 P

### 1.1 基本想法：分块在片上算

FA1 将 `Q` 切成 query tile，将 `K, V` 切成 key/value tile。数学上，可以把一个 `Q_i` 的输出理解成依次合并 `(K_1, V_1)`, `(K_2, V_2)`, ... 的贡献。

原始 FA1 的参考调度为了复用当前 `K_j, V_j` tile，采用 **KV 外层、Q 内层**：

```text
for K_j, V_j:
    将 K_j, V_j 读入 SRAM
    for Q_i:
        读入 Q_i 以及当前 O_i、m_i、ell_i
        用 K_j, V_j 更新这组状态
        写回更新后的 O_i、m_i、ell_i
```

因此，FA1 不会把完整 `N × N` 的 `S, P` 写入 HBM，但同一个 `Q_i` 的累计状态仍会在多个 KV tile 之间往返于 HBM。FA2 正是通过翻转这层循环、按 Q tile 分配 worker 来进一步减少这些往返。

关键不是少算 `Q @ K.T`；FA1 仍然是精确的、计算量为 `O(N²d)` 的 attention。它首先省掉的是完整 `N × N` 的 `S, P` 的 HBM 读写。

### 1.2 为什么可以分块做 softmax：online softmax

对一行 score `s_1, ..., s_N`，数值稳定 softmax 可写为：

```text
m   = max(s_j)
ell = sum(exp(s_j - m))
O   = sum(exp(s_j - m) * V_j) / ell
```

扫描 KV tile 时，维护每个 query 行的三个状态：

```text
m       当前为止的最大 score
ell     当前为止的 softmax 分母
O_tilde 当前为止、尚未除以分母的输出分子
```

若新 tile 使最大值从 `m_old` 变为 `m_new`，先计算：

```text
alpha = exp(m_old - m_new)
```

旧状态换到新尺度后，可与新 tile 合并：

```text
ell_new = alpha * ell_old
          + sum(exp(s_j - m_new), j in new_tile)

O_tilde_new = alpha * O_tilde_old
              + sum(exp(s_j - m_new) * V_j, j in new_tile)
```

处理完全部 KV tile 后：

```text
O = O_tilde / ell
```

因此，tile 的先后顺序不会改变最终数学结果。

### 1.3 backward：用重算换显存

反向传播需要概率 `P`，但保存完整 `P` 仍是 `N²` 级别的显存。FA1 不保存它；forward 只保存较小的行级统计量（如 log-sum-exp）和输出，backward 读取一个 tile 后重新算对应的 score/概率，再立刻计算梯度。

```text
少保存巨大激活 → backward 多做部分计算 → HBM 读写显著更少
```

这就是 **recomputation / rematerialization**。在 GPU 上，额外算术经常比搬运大矩阵更便宜。

---

## 2. FlashAttention-2 / FA2：相同数学，更多有效并行

FA2 仍然是精确 attention，也仍使用 tiling、online softmax 和 backward recomputation。它主要改进任务划分与 kernel 内部的数据流。

### 2.1 不在每个 KV tile 都做完整归一化

FA1 可以把每轮的中间输出保持为已归一化状态：

```text
O = O_tilde / ell
```

这样下一轮来时，需要先乘回 `ell` 恢复分子、合并新贡献，再除以新的 `ell`。

FA2 改为始终维护 `O_tilde` 与 `ell`：

```text
每个 tile：更新 O_tilde（分子）和 ell（分母）
最后一次 ：O = O_tilde / ell
```

注意：FA2 **不是**不更新分母，也不是不做因最大值变化而必须的缩放；它只是把反复的输出归一化除法推迟到最后。

FA2 还保存每行的 `L`：

```text
L = m + log(ell)
  = log(sum(exp(s_j)))
```

于是 backward 重算概率时可直接写成：

```text
P_ij = exp(S_ij - L_i)
```

无需分别读取 `m_i` 与 `ell_i`，也无需显式做一次除法。

### 2.2 增加 sequence-length 维度的并行

FA1 的并行度主要来自 batch 和 head。若 batch 小、head 少、但序列很长，GPU 的许多 SM 可能空闲。

FA2 在 forward 额外按 **query 行块** 并行：

```text
Worker 1：负责 Q 的第 1 个行块，遍历全部 KV tile
Worker 2：负责 Q 的第 2 个行块，遍历全部 KV tile
...
```

不同 query 行得到不同输出行，彼此不需要通信。

backward 的划分则更适合按 **KV 列块**：

```text
dV_j = sum(P_ij.T @ dO_i, over all query rows i)
dK_j = sum(dS_ij.T @ Q_i, over all query rows i)
```

`dK_j, dV_j` 都是沿 query 行求和，因此一个 worker 独占一列 KV block 时能在片上累积它们。代价是：

```text
dQ_i = sum(dS_ij @ K_j, over all KV columns j)
```

会被多列 worker 共同贡献，FA2 对它使用 atomic add。

#### QUESTION 1 ：为什么 forward 按照 query，backward 更适合按 KV 列块呢？
```text
因为 forward 的输出按 query 行独立；而 backward 中 `K, V` 的梯度按 KV 列累加。

P 的行：query token i
P 的列：key/value token j
```

forward 时：

```text
O_i = sum(P_ij * V_j, over all j)
```

固定一行 `Q_i` 后，worker 只需遍历所有 KV，最终只写自己的 `O_i`。不同 query 行写的是不同输出行，不会冲突，所以按 query 行块并行最自然。

```text
Worker 1 → Q 行块 1 → O 行块 1
Worker 2 → Q 行块 2 → O 行块 2
```

backward 时，关键梯度是：

```text
dV_j = sum(P_ij.T @ dO_i, over all query rows i)
dK_j = sum(dS_ij.T @ Q_i, over all query rows i)
dQ_i = sum(dS_ij @ K_j, over all KV columns j)
```

`dK_j`、`dV_j` 都需要汇总**所有 query 行**对第 `j` 个 KV 的贡献。

例如：

```text
dV_1 = P_11.T @ dO_1 + P_21.T @ dO_2 + ...
```

若 backward 仍按 query 行分 worker：

```text
row worker 1 要更新 dV_1、dK_1
row worker 2 也要更新 dV_1、dK_1
```

它们会同时写同一份 `dV_1, dK_1`，需要大量 atomic add 或额外归约。

因此 FA2 的 backward 改为按 KV 列块：

```text
KV worker 1：遍历所有 Q 行，独占累积 dK_1、dV_1
KV worker 2：遍历所有 Q 行，独占累积 dK_2、dV_2
```

这样 `dK, dV` 不需要跨 worker 合并。代价是多个 KV worker 会共同更新同一个 `dQ_i`，所以 FA2 接受对 `dQ` 做 atomic add。

一句话：

```text
forward：O 按 query 行独立 → 按 Q 行并行
backward：dK、dV 按 KV 列汇总 → 按 KV 列并行
```

这是在“三份梯度里，尽量只让一份需要跨 worker 合并”的折中。


### 2.3 CTA 内避免 Split-K：按 Q 分 warp

这和上面的「多个 thread block 如何分工」不同；这里讨论的是**一个 thread block 内的 4 或 8 个 warp**。

```text
FA1：不同 warp 切分 K/V
     → 每个 warp 都算同一批 query 的局部输出
     → 必须写 shared memory、同步、再归约

FA2：不同 warp 切分 Q 的行
     → 每个 warp 算自己 query 行的完整输出
     → 不需要跨 warp 合并输出
```

因此 FA2 减少 shared-memory 读写和 warp 同步，也提高了长序列、小 batch 场景的 SM 占用率。

---

## 3. FlashAttention-3 / FA3：让 Hopper 的不同硬件同时工作

FA3 的核心并不是继续改变 online softmax，而是为 NVIDIA Hopper（尤其 H100）重写 kernel 调度。FA2 已经减少了大部分 HBM 往返，但在 H100 上仍没有充分利用新的异步硬件能力。

### 3.1 TMA + WGMMA：搬运与计算解耦

Hopper 提供两项关键能力：

- **TMA（Tensor Memory Accelerator）**：异步搬运 HBM 与 shared memory 之间的 tile；
- **WGMMA（Warpgroup Matrix Multiply-Accumulate）**：由一个 warpgroup 发起、更高吞吐的异步 Tensor Core 矩阵乘。

FA3 做了 warp specialization：

```text
producer warps：发起 TMA，预取下一块 K/V
consumer warps：用 WGMMA 计算当前块的 QK^T 和 P@V
```

于是「当前 tile 在计算」和「下一 tile 在搬运」可以重叠。这里的异步不违反数据依赖：它只是让**下一块数据的加载**与**上一块的计算**同时进行。

### 3.2 GEMM–softmax 流水线

一个 tile 内部的依赖仍然存在：

```text
QK^T → softmax → P@V
```

同一 tile 不能乱序，但跨 tile 可以做两级流水：当一个 warpgroup 在计算某 tile 的 softmax 时，另一个 warpgroup 做别的 tile 的矩阵乘；之后角色交换（ping-pong）。

```text
warpgroup A：softmax(tile j)
warpgroup B：GEMM(tile j+1) 或 P@V(已就绪 tile)
```

这样 Tensor Cores 做矩阵乘时，负责 `exp` 等特殊函数的硬件单元也能工作。这个重叠很重要：H100 的 Tensor Core 极快，而 softmax 中的指数运算相对慢，串行执行会让其中一类硬件闲置。

### 3.3 FP8：更高吞吐与更低量化误差

FP8 的 Tensor Core 吞吐更高，但动态范围更窄。FA3 的 FP8 路径采用：

- **block quantization**：每个 tile 使用自己的 scale，避免少数 outlier 支配整张张量的量化范围；
- **incoherent processing**：对 `Q, K` 施加带随机符号的 Hadamard/正交变换，将 outlier 分散到多个维度。

若 `M` 是正交矩阵，则理想数学下：

```text
(Q @ M) @ (K @ M).T
= Q @ M @ M.T @ K.T
= Q @ K.T
```

因此可保持注意力 score 不变，同时让量化前的数值分布更均匀。FP8 是低精度路径；FP16/BF16 路径仍是精确 FlashAttention。

### 3.4 FA3 的适用边界

FA3 的主要收益来自 Hopper 的 TMA、WGMMA 与 FP8 硬件，不应把论文中 H100 的速度提升直接套到 A100。最终论文报告，H100 上 BF16 最高可达 840 TFLOPs/s（85% 理论峰值利用率），相对 FA2 约快 `1.5–2.0×`；FP8 路径可达约 1.3 PFLOPs/s。[FA3 论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)

---

## 4. FlashDecoding：decode 时改为沿 KV cache 切分

### 4.1 为什么 FA2 的 query 并行在 decode 时不够

decode 每一轮通常只处理一个新 token：

```text
Q                shape: B × H × 1 × d
K_cache, V_cache shape: B × H × T × d
```

`T` 是已有上下文长度，可能是 32K、128K 甚至更长。

FA2 的 forward 主要沿 query 行分块；但 decode 中每个请求只有一行 query。若 batch 又小，则可启动的 worker 数可能远少于 GPU 的 SM 数：少数 worker 串行扫完整个长 KV cache，而大量 SM 空闲。

```text
prefill：Q 长 → 沿 Q 切分足以产生很多任务
decode ：Q = 1 → 沿 Q 切分几乎没有额外任务
```

### 4.2 Split-KV：把一条长 KV cache 分给多个 worker

FlashDecoding 将 KV cache 沿序列维度切成 `R` 段：

```text
KV cache: [split 1] [split 2] [split 3] ... [split R]
             ↓           ↓           ↓
           worker 1    worker 2    worker 3
```

每个 worker 独立对自己的 KV 段运行小型 FlashAttention，并输出三个局部状态：

```text
m_r       = max(s_j, j in split r)
ell_r     = sum(exp(s_j - m_r), j in split r)
O_tilde_r = sum(exp(s_j - m_r) * V_j, j in split r)
```

第二个很小的 combine/reduction kernel 再将这些分段结果精确合并：

```text
m       = max(m_r, over all splits r)
ell     = sum(exp(m_r - m) * ell_r, over r)
O_tilde = sum(exp(m_r - m) * O_tilde_r, over r)
O       = O_tilde / ell
```

这本质上就是把 FA1 的 online softmax 合并公式，从「同一 worker 内的多个 KV tile」推广到「多个 worker 各自处理的一段 KV cache」。

若每个 split 保存局部归一化结果 `O_r = O_tilde_r / ell_r` 和：

```text
L_r = m_r + log(ell_r)
```

则 combine 还能简洁地写成：

```text
L = log(sum(exp(L_r), over r))
O = sum(exp(L_r - L) * O_r, over r)
```

### 4.3 为什么它适合长上下文 decode

FlashDecoding 用一次很小的最终归约，换取更多并行任务：

```text
原先：1 个 worker 顺序扫描整条 KV cache
现在：R 个 worker 并行扫描 R 段 KV cache，再做一次 combine
```

它仍不物化完整 attention matrix，因此保留 FlashAttention 的低额外显存特性；但在 query 很短、KV 很长、batch 很小的时候，可以显著提高 SM 利用率。作者报告在长上下文生成场景中端到端最高可达 8× 加速；性能收益依赖模型、上下文长度、batch 和 GPU。[Flash-Decoding 作者文章](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

### 4.4 它不是任何场景都更快

Split-KV 会产生局部输出/LSE 临时状态，并增加一个 combine kernel。因此：

- **长 KV、短 Q、小 batch**：通常值得 split；
- **短上下文或 batch 已足够大**：额外 reduction 可能得不偿失；
- **数值**：数学上等价，但浮点加法不满足结合律，不同合并顺序可能带来极小的数值差异。

实际系统通常根据 sequence length、batch、head 数和 SM 数自适应选择 split 数。




## 5. 一张表记住差异

| 方法 | 主要场景 | 核心问题 | 新增的并行/流水维度 | 关键代价 |
|---|---|---|---|---|
| FA1 | 训练、prefill | `S, P` 的 HBM 往返 | tile 内流式 KV | backward 需要重算 |
| FA2 | 训练、prefill | SM 占用低、warp 通信多 | Q 行块并行；warp 内切 Q | backward 对 `dQ` 有原子累加 |
| FA3 | Hopper 上的训练、prefill | 搬运、GEMM、softmax 串行等待 | TMA/WGMMA 异步流水 | 更依赖 Hopper，寄存器压力更高 |
| FlashDecoding | 长上下文 decode | `Q` 长度为 1，Q 维度无并行 | KV cache sequence 维度 split | 多一步 combine/reduction |

---

## 6. 最后用一句话串起来

```text
FA1：让 attention 少写大矩阵。
FA2：让更多 SM 和 warp 做有用的工作。
FA3：让 H100 的搬运、Tensor Core 和 softmax 单元同时做事。
FlashDecoding：当 Q 只有一个时，改切长 KV cache 来制造并行度。
```

## 参考资料

1. [FlashAttention（FA1）论文](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
3. [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608) 与 [作者图解](https://tridao.me/blog/2024/flash3/)
4. [Flash-Decoding 作者文章](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
5. [FlashDecoding++ 论文](https://arxiv.org/abs/2311.01282)
6. [FA1/FA2/FA3](https://zhuanlan.zhihu.com/p/668888063)、[FlashDecoding/FlashDecoding++](https://zhuanlan.zhihu.com/p/696075602)
