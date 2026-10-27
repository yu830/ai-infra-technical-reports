# FlagOS Task 156：分段 LoRA K 投影的 Triton 实现

## 摘要

Task 156 的 step_a_q 算子面向 absorbed-MLA 的 LoRA 修正：按请求所属 segment 选择 adapter，再对每个 attention head 的输入与该 adapter 的 K 部分做矩阵乘。实现采用 segment/head 分块，在特定短 decode 形状下组合多个 head，并为昆仑芯提供按数据布局选择的实现。

2026 年 10 月 22 日的 FlagOS 题目榜单记录中，队伍 YuJiacheng 位列该题第 3，平台显示平均加速比 **10.74×**。下文区分这条榜单记录和可以读取源码的提交包，便于理解数字与实现各自的证据范围。

## 问题与计算契约

输入 q_nope 的形状为 [S, H, K]，其中 S 是 token 数、H 是 head 数；权重 B_buf 为 [adapter 数, H × full_K_per_head, R]，R 为固定 LoRA rank。输出形状为 [S, H, R]。

每个 segment 通过 weight_indices 选择 adapter；可选 permutation 将 segment 内位置映射回原始 token 行。对每个有效 segment、token 行 i 和 head h，计算：

    out[i, h, :] = q_nope[i, h, :] @
                   B_buf[adapter, h × full_K_per_head : h × full_K_per_head + K, :]

每个 head 的 K/V 权重共同占据 full_K_per_head 行，本算子仅读取前 K 行。空 segment 与零 rank adapter 跳过；未覆盖的输出行保持为零。参考实现以 FP32 进行矩阵乘，再转换为输入 dtype；题目给出的 BF16 容差为 atol=rtol=0.015。

## 实现

### 通用路径

以 segment、head 和输出列为分派维度，将 token 行与 rank 列构成输出 tile，沿 K 维分块归约。候选中的 BM 按 segment 长度选择 16 或 64，BN 按 rank 选择 16、32 或 64，BT 为 64；Triton dot 使用 IEEE 输入精度和 FP32 累加。输入及权重读取使用实际 stride，因而允许非连续布局。

当最长 segment 不超过 4、H 至少为 8、S×H 至少为 128、16 < R ≤ 32 且 K ≤ 256 时，候选使用短 decode 路径：每个程序处理一个 token，并组合 4 个 head 完成计算。该路径针对较短 segment 减少需要分别调度的 head 工作单元；其收益仍依赖实际形状和后端。

输出初始化通常由独立 Triton kernel 完成；在未使用 permutation、非短路径及其他条件满足时，候选将部分清零与分段投影合并。metadata 每次从当前 batch 读取，不跨调用缓存 adapter 或输入内容。

### 昆仑芯路径

对于无 permutation 的输入，候选使用直接分块投影，并保留实际 q_nope / B_buf stride。存在 permutation 时，路径将 [head, rank] 展平成一维输出位置，通过四组 FP32 累加器显式遍历 K。该选择避免对重排行直接使用矩阵 gather，代价是这一路径的权重复用方式与矩阵 tile 不同。

## 平台结果与版本

题目榜单于 2026-10-22 20:14 记录的队伍结果如下；这些数值是平台公布的赛题指标，具体 case 形状和逐 case 时间未公开。

| 平台设备标签 | 加速比 |
|---|---:|
| 天数 | 9.73× |
| 沐曦 | 12.11× |
| 海光 | 27.56× |
| 昆仑芯 | 1.71× |
| 华为 | 2.71× |
| 国际通用芯片 A | 13.40× |
| 国际通用芯片 B | 7.96× |

榜单显示平均加速比 10.74142857×，排名分数为 1.2，题目名次为第 3。排名按各芯片名次计算，不应仅以平均加速比推算。

本文读取的源码包为 2026-10-22 16:35 的候选，源码提交标识为 13721f12c6e9，ZIP 的 SHA-256 为 cfeabdae2429744f0bfcde72e6530934c3d61b653c0f7f91c410b76f48d2c1f3e。对应的用户提供平台回执显示平均加速比 **10.73×**。它与稍后榜单的 **10.74×** 是两条独立记录；目前不能把榜单数字绑定为该 ZIP 的精确成绩，也不能确认平台实际加载的源码哈希。

## 开发中的取舍

早期独立的一维候选在一次平台反馈中满足正确性，但昆仑芯加速比低于 0.01×，未达到该题 0.1× 的性能门槛。另一条采用打包与矩阵路径的候选在真实目标评测中出现数值不匹配和非法内存访问。两者属于不同版本；这些反馈促使后续实现按布局区分直接矩阵路径与重排路径，而不能作为本文候选的逐项性能归因。

题目公布的 PyTorch 参考代码包含主机 metadata 读取和启动开销。平台平均加速比是算子赛题结果，不等同于相对 SGLang 生产路径的端到端推理加速。

## 来源与署名

- [FlagOS Task 156 官方题目与参考实现](https://flagos.io/flagos/api/v1/races/782kzq4m/operator-tasks/step_a_q)
- [FlagGems-sglang 项目](https://github.com/flagos-ai/FlagGems-sglang)
- 榜单记录与候选包的原始材料分别保留；候选源码文件带有 FlagOS Contributors 版权及 Apache-2.0 标识。
- 本报告由 OpenAI Codex 协助整理；源码和其他贡献的原有署名及披露保持不变。
