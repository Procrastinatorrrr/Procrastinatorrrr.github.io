---
title: "从 SFT 到 OPD：训练分布与推理分布的对齐之路"
date: 2026-04-26
author: "Yijun Long"
tags: ["SFT", "Knowledge Distillation", "On-Policy", "LLM", "OPD"]
categories: ["Machine Learning", "LLM", "Knowledge Distillation"]
description: "从数学原理出发，推导从监督微调到离线蒸馏再到在线蒸馏的演进逻辑。"
math: true
summary: "从数学原理出发，推导从监督微调到离线蒸馏再到在线蒸馏的演进逻辑。"
weight:
slug: "202604-sft-to-distillation"
draft: false
comments: true
showToc: true
TocOpen: true
autonumbering: true
hidemeta: false
disableShare: true
searchHidden: false
showbreadcrumbs: true
mermaid: true
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

> 本文假设读者了解大语言模型的基本训练流程（预训练-SFT-RL），但不要求预先掌握知识蒸馏的背景知识。

## 1. SFT 与人工标注的局限

### 1.1 SFT 的数学目标

监督微调（Supervised Fine-Tuning, SFT）是大语言模型训练管线中最基础的阶段。给定一组（指令，回答）对 $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$，SFT 的目标是让模型学会在给定用户输入 $x$ 的条件下，逐 token 地生成正确的回答 $y$。其训练损失为标准的自回归交叉熵：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{\lt t}) \right]
$$

其中 $\pi_\theta$ 是参数为 $\theta$ 的语言模型，$y_{\lt t}$ 表示真实回答 $y$ 中位置 $t$ 之前的所有 token。这个损失函数的含义是：在给定上文 $(x, y_{\lt t})$ 的条件下，最大化正确 token $y_t$ 的对数概率。训练时，模型每一步都以**真实答案的前缀** $y_{\lt t}$ 作为输入——这种策略被称为**教师强制（Teacher Forcing）**。

### 1.2 人工标注的数据瓶颈

SFT 效果的上限取决于训练数据的质量，而高质量的人工标注数据面临三重瓶颈：

**成本瓶颈**。构建一套高质量的指令微调数据集需要大量领域专家参与。人类标注员不仅要理解用户的意图，还要写出逻辑严谨、表达准确的回答。对于数学推理、代码生成等专业领域，标注成本尤为高昂。

**规模瓶颈**。人类能产出的数据量级有限——即使投入大量人力，数据集的规模也很难突破百万量级。而预训练阶段使用的 token 数量通常是万亿级别，两者之间存在巨大的鸿沟。模型在 SFT 阶段能接触到的指令空间和回答模式只是真实世界中极小的一部分。

**能力瓶颈**。这是最根本的限制：SFT 的训练目标是拟合人类标注员的回答，因此模型的能力上限被标注者本人的水平所约束。无论模型的参数量多大、预训练数据多丰富，它都无法在 SFT 阶段学会标注者不掌握的知识或技能。一个不会证明数学定理的标注员，不可能通过标注教会模型证明定理。

> 简言之，SFT 让模型学会了"模仿人类"，但无法让模型"超越人类"。模型能力的上限被锁定在标注者的水平上。

## 2. 教师合成数据：蒸馏的第一步

### 2.1 用更强的模型替代人工标注

面对人工标注的瓶颈，一个极为自然的思路是：既然模型已经通过预训练获得了强大的语言能力，那为什么不直接用一个更强大的**教师模型**来生成训练数据，替代昂贵的人工标注？

具体做法是：

1. 教师模型对大量 prompt $x$ 生成回答 $y \sim p_{\text{Teacher}}(\cdot \mid x)$
2. 将这些 $(x, y)$ 对保存为**静态数据集**
3. 在这个数据集上对学生模型进行**标准 SFT 训练**——使用交叉熵损失

$$
\mathcal{L}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{Teacher}}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{\lt t}) \right]
$$

注意，这里的损失函数与标准 SFT **完全相同**——仍然是交叉熵，仍然是在 one-hot 的采样 token 上最大化对数概率。唯一的变化是数据的来源：从人类标注变为教师模型生成。

这个简单的方法在实践中取得了巨大成功。大量被归类为"SFT"的工作本质上就是这种"教师合成数据 + 标准交叉熵"的范式：Alpaca 用 GPT-3.5 生成 52K 指令数据微调 LLaMA 7B、Vicuna 用 ChatGPT 对话数据微调 LLaMA 13B、DeepSeek-R1 用 671B 推理模型的输出微调密集模型——它们的训练损失都是交叉熵，但训练数据全部来自更强的教师模型。

### 2.2 One-Hot 信号的信息损失

然而，交叉熵损失有一个被忽略的问题。让我们仔细审视这个损失函数在做什么：

$$
\mathcal{L}(\theta) = -\log \pi_\theta(y_t \mid x, y_{\lt t})
$$

它将教师模型的输出**压缩为一个 one-hot 向量**——只有被采样的那个 token $y_t$ 得到信号（目标概率为 1），词表中其他所有 token 的目标概率都被置为 0。教师的采样过程 $y_t \sim p_{\text{Teacher}}(\cdot \mid x, y_{\lt t})$ 从一个完整的概率分布中只取了一个样本，而这个分布中蕴含的丰富信息在 one-hot 化的过程中被丢弃了。

用一个具体的例子来说明。假设教师模型在面对"推导勾股定理"这个问题时，在某一步输出的概率分布为：

$$
p_{\text{Teacher}} = \{ \text{"因此"}: 0.45, \text{"所以"}: 0.30, \text{"于是"}: 0.15, \text{"故"}: 0.08, \text{其他}: 0.02 \}
$$

教师采样得到"因此"后，交叉熵损失告诉学生："在这个位置，正确答案是'因此'，其他所有 token 的目标概率都是 0。" 但教师的分布实际上在传递更丰富的信息："因此"最好（0.45），"所以"也不错（0.30），"于是"勉强可以（0.15），"故"偏弱（0.08）。这些** token 之间的相对关系**——Hinton 称之为**暗知识（Dark Knowledge）**——在 one-hot 化后完全丢失了。

> 换句话说，交叉熵在告诉学生"老师选了什么"，而 KL 散度能告诉学生"老师是怎么想的"。

这个问题在低熵的位置（即教师非常确定的位置）影响不大，但在高熵的位置（即教师本身就不确定、多个 token 都合理的位置）影响显著——而这些位置往往对应着推理过程中的关键决策点。

## 3. 离线蒸馏：学分布而非学 one-hot

### 3.1 从交叉熵到 KL 散度

上一节的分析指向一个自然的改进方向：不要让学生只学习教师采样的 one-hot token，而是让学生**匹配教师的完整输出分布**。这正是知识蒸馏的核心思想。

给定教师的 logits $z_T$ 和学生的 logits $z_S$，在温度 $\tau$ 下软化后的分布分别为：

$$
p_T = \text{softmax}(z_T / \tau), \quad p_S = \text{softmax}(z_S / \tau)
$$

与交叉熵不同，知识蒸馏使用 **KL 散度**来衡量两个分布之间的差异。KL 散度是不对称的——两种参数顺序对应不同的优化行为。在离线蒸馏的场景下，标准的 KL 散度选择是 **forward KL**：

$$
\mathcal{L}_{\text{FKL}} = \text{KL}(\pi_T \| \pi_\theta) = \sum_{v \in \mathcal{V}} \pi_T(a_v \mid s) \log \frac{\pi_T(a_v \mid s)}{\pi_\theta(a_v \mid s)}
$$

其中 $\pi_T$ 是教师分布，$\pi_\theta$ 是学生分布，$s$ 是前缀。注意 KL 散度的第一个参数是教师——这是一个关键的选择，与离线蒸馏中"序列由教师采样"这一事实紧密耦合。

由于教师分布 $\pi_T$ 不依赖于学生参数 $\theta$，forward KL 可以写成：

$$
\mathcal{L}_{\text{FKL}} = \underbrace{\sum_{v} \pi_T(a_v \mid s) \log \pi_T(a_v \mid s)}_{\text{与 } \theta \text{ 无关的常数}} - \sum_{v} \pi_T(a_v \mid s) \log \pi_\theta(a_v \mid s)
$$

因此，最小化 forward KL 等价于最小化**教师软标签下的交叉熵**——它与 SFT 的交叉熵在数学结构上完全一致，只是目标分布从 one-hot 变成了教师的完整分布 $\pi_T$。

### 3.2 Forward KL 的梯度：逐坐标校正

Forward KL 对 student logit $z_j$ 的梯度有一个非常经典且简洁的结果：

$$
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j} = p_j - q_j
$$

其中 $p_j = \pi_\theta(a_j \mid s_t)$，$q_j = \pi_T(a_j \mid s_t)$。梯度下降更新为 $z_j \leftarrow z_j - \eta(p_j - q_j)$，这意味着：

- 若 $p_j > q_j$（student 高估了该 token），梯度为正，梯度下降会**压低**其 logit
- 若 $p_j < q_j$（student 低估了该 token），梯度为负，梯度下降会**抬高**其 logit
- 若 $p_j = q_j$，达到局部匹配，不再产生梯度

这说明 forward KL 的行为是一种**逐坐标的分布校正（mode-covering）**：对每个 token 独立地把 student 的概率往 teacher 的对应值上拉。这种行为直觉上非常自然——teacher 给得比 student 高，就抬；teacher 给得比 student 低，就压。

### 3.3 离线蒸馏的训练流程

综合以上分析，**离线蒸馏（Off-Policy Distillation，也称 SeqKD）** 的完整流程为：

1. 教师模型对大量 prompt $x$ 生成回答 $\tau \sim \pi_T(\cdot \mid x)$，保存为静态数据集
2. 对数据集中的每条序列，在每个 token 位置计算学生分布与教师分布之间的 forward KL
3. 最小化该 forward KL

与上一节"教师合成数据 + 交叉熵"的方案相比，离线蒸馏用教师软标签替代了 one-hot 标签，保留了教师输出分布中的暗知识。从"学 one-hot"到"学分布"，这是蒸馏真正区别于普通 SFT 的标志。

### 3.4 暴露偏差：训练分布 ≠ 推理分布

离线蒸馏虽然解决了"学什么"的问题（从 one-hot 到分布），但它没有解决"在谁的分布上学"的问题。无论是在 SFT 中用人工数据，还是在离线蒸馏中用教师预生成的数据，训练时模型面对的状态始终由**外部数据**决定，而非由模型自身的策略决定。

将自回归生成形式化为 **马尔可夫决策过程（MDP）** 可以更精确地刻画这个问题：

- **状态空间** $\mathcal{S}$：所有可能的 token 前缀 $s_t = (x, y_{\lt t})$
- **动作空间** $\mathcal{A}$：词表 $\mathcal{V}$ 中的每一个 token
- **状态转移**：确定性的，即选择动作 $y_t$ 后状态变为 $s_{t+1} = (x, y_{\lt t}, y_t)$

训练时，模型访问的状态分布是 $d_{\mathcal{D}}(s)$——由数据集中的真实 token 序列（人工标注或教师生成的）决定。推理时，模型必须自回归生成，每一步以自身前一步的输出为输入，因此状态访问分布变为 $d_{\pi_\theta}(s)$——由模型自身的策略诱导。当模型不完美时：

$$
d_{\mathcal{D}}(s) \neq d_{\pi_\theta}(s)
$$

这就是**暴露偏差（Exposure Bias）**。对于 $d_{\pi_\theta}(s) \setminus d_{\mathcal{D}}(s)$ 中的状态——即模型自己"走偏"后产生的中间前缀——模型在训练中从未见过，对它们的行为完全不可预测。

### 3.5 误差累积：$O(\varepsilon \cdot T^2)$

暴露偏差的直接后果是**误差累积（Error Compounding）**。根据 DAgger（Ross et al., 2011）的理论分析，假设模型在训练分布下的每步误差有界：

$$
\mathbb{E}_{s \sim d_{\mathcal{D}}} \left[ \mathbb{I}(\pi_\theta(s) \neq \pi^*(s)) \right] \leq \varepsilon
$$

那么在模型自身分布下的总误差满足：

$$
\mathbb{E}_{s \sim d_{\pi_\theta}} \left[ \sum_{t=1}^{T} \mathcal{L}(s_t) \right] \leq O(\varepsilon \cdot T^2)
$$

> 直觉：一旦模型在某一步 $t$ 犯了一个小错，第 $t+1$ 步的输入就会偏离训练分布，使得模型在第 $t+1$ 步更容易犯错；这个新错误进一步偏移第 $t+2$ 步的输入……误差以**二次速度**增长。对于需要长链推理的任务（如数学证明、代码生成），$T$ 可以达到数百甚至数千，$T^2$ 的累积意味着模型在生成后半段几乎必然偏离正确轨道。

将此结论应用到 SFT 与离线蒸馏中，可知二者面临相同的暴露偏差和相同的 $O(\varepsilon \cdot T^2)$ 误差上界。用教师数据替代人工数据、用 KL 散度替代交叉熵，这些改进都没有触及问题的根源——训练分布与推理分布之间的不一致。只要训练数据是预生成且固定的，无论数据质量多高、损失函数多精细，暴露偏差都无法消除。

## 4. 在线蒸馏：在学生自身的分布上学习

### 4.1 核心思想：训练分布 = 推理分布

上一节的结论表明，暴露偏差的根源在于 $d_{\mathcal{D}} \neq d_{\pi_\theta}$。在线蒸馏（On-Policy Distillation, OPD）的出发点极为直接：**让训练分布等于推理分布**。

具体做法：

1. 学生模型自己生成回答 $y \sim p_\theta(\cdot \mid x)$
2. 教师模型对学生生成的序列提供逐 token 的 logits 反馈 $p_{\text{Teacher}}(y_t \mid x, y_{\lt t})$
3. 在学生自己采样的序列上，用 KL 散度最小化教师与学生之间的差异

此时：

$$
\underbrace{y \sim p_\theta(\cdot \mid x)}_{\text{训练分布}} \quad \Longleftrightarrow \quad \underbrace{y \sim p_\theta(\cdot \mid x)}_{\text{推理分布}}
$$

训练分布与推理分布**完全一致**，暴露偏差被消除。

### 4.2 从 $O(T^2)$ 到 $O(T)$

DAgger 算法（Ross et al., 2011）的核心思想是：不是让学生在专家的策略分布 $d_{\pi^*}$ 上训练，而是让学生**在自己的策略分布 $d_{\pi_\theta}$ 上运行**，然后在它实际访问的状态上向专家请求最优动作，此时总误差为 $O(\varepsilon \cdot T)$。

这正是在线蒸馏在做的事情：在 $d_{\pi_\theta}$（学生自身的轨迹）上训练，由教师在 $d_{\pi_\theta}$ 访问的状态上提供监督。因此，根据 DAgger 的理论，误差变为误差 $O(\varepsilon \cdot T)$。

> 直觉：在线蒸馏中，学生生成的每一条"错误轨迹"都会得到教师的反馈——模型因此学会了**如何从自己犯的错中恢复**。这些恢复能力在推理时至关重要：即使模型在某一步走偏，它也曾在训练中见过类似的偏移状态，知道如何纠正。

### 4.3 OPD 的反向 KL 与两种梯度实现

在线蒸馏的理论目标是最小化学生与教师之间的**逐 token KL 散度**：

$$
\mathcal{L}_{\text{OPD}}(\theta) = \sum_{t=1}^{|y|} \text{KL}\left( \pi_\theta(\cdot \mid s_t) \| \pi_T(\cdot \mid s_t) \right)
$$

其中 $s_t = (x, a_1, \ldots, a_{t-1})$ 是前缀序列，$a_t$ 是实际采样出的 token。这里 KL 散度的方向是 $\pi_\theta \| \pi_T$（reversed KL），而第 3 章的离线蒸馏使用的是 $\pi_T \| \pi_\theta$（forward KL）。这是因为，根据定义，KL 散度第一个参数 $P$ 是真实分布（也即采样的分布）：

$$
\text{KL}(P \| Q) = \mathbb{E}_{a \sim P}\left[\log \frac{P(a)}{Q(a)}\right]
$$

在离线蒸馏中，序列由教师采样（$a \sim \pi_T$），因此使用 forward KL $\text{KL}(\pi_T \| \pi_\theta)$；在在线蒸馏中，序列由学生采样（$a \sim \pi_\theta$），因此自然使用反向 KL $\text{KL}(\pi_\theta \| \pi_T)$。

将单步反向 KL 展开：

$$
\ell_t = \text{KL}(\pi_\theta \| \pi_T) = \sum_{v \in \mathcal{V}} \pi_\theta(a_v \mid s_t) \log \frac{\pi_\theta(a_v \mid s_t)}{\pi_T(a_v \mid s_t)}
$$

下一个问题是：**如何对这个目标求梯度？** 这里有两种不同的实现路径。

#### GKD-style：Full-Vocabulary Exact Gradient

直接对上式求导，可得：

$$
\nabla_\theta \ell_t = \sum_{v \in \mathcal{V}} \left( \log \frac{\pi_\theta(a_v \mid s_t)}{\pi_T(a_v \mid s_t)} + 1 \right) \nabla_\theta \pi_\theta(a_v \mid s_t)
$$

这个梯度的核心特征是**对词表中的所有 token 都产生梯度信号**——每一步更新都利用了教师在整个词表上的分布信息。

**Logit 角度的梯度**。上述参数梯度可以进一步简化为对单个 logit $z_j$ 的梯度。由于教师分布 $q_j$ 不依赖学生参数 $\theta$，利用 softmax 的 Jacobian $\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$：

$$
\begin{aligned}
\frac{\partial \ell_t}{\partial z_j} &= \sum_{k} \frac{\partial p_k}{\partial z_j} \cdot \frac{\partial \ell_t}{\partial p_k} \\
&= \sum_{k} p_k(\delta_{kj} - p_j) \left( \log \frac{p_k}{q_k} + 1 \right) \\
&= p_j \left( \log \frac{p_j}{q_j} + 1 \right) - p_j \sum_{k} p_k \left( \log \frac{p_k}{q_k} + 1 \right) \\
&= p_j \left( \log \frac{p_j}{q_j} + 1 \right) - p_j \left( \text{KL}(p \| q) + 1 \right) \\
&= p_j \left( \log \frac{p_j}{q_j} - \text{KL}(p \| q) \right)
\end{aligned}
$$

其中 $p_j = \pi_\theta(a_j \mid s_t)$，$q_j = \pi_T(a_j \mid s_t)$，$\delta_{kj}$ 是 Kronecker delta。这个式子表明：反向 KL 并非简单地逐 token 拉齐（"student 高于 teacher 就压低，低于 teacher 就抬高"），而是将每个 token 的局部不匹配程度与全局平均不匹配程度做比较。它会强烈打压 student 错押而 teacher 不认可的高概率模式，并把概率重新分配到 teacher 更支持的候选上——这正是 reverse KL 被称为 **mode-seeking** 的数学根源。

作为对比，离线蒸馏中常用的 forward-KL 的 logit 梯度则简单得多：

$$
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j} = p_j - q_j
$$

Forward-KL 的行为非常直接：逐 token 地把 student 的概率往 teacher 的对应值上拉，是一种**逐坐标的分布校正（mode-covering）**。两者在优化偏好上的差异可以通过一个例子直观理解：假设 $p = (0.2, 0.3, 0.5)$，$q = (0.7, 0.2, 0.1)$。对于 $t_2$（student 概率 0.3，teacher 概率 0.2），forward-KL 会压低它（因为 $p_2 > q_2$），但 reverse-KL 反而会**抬高**它——因为当前 student 最大的问题是对 $t_3$ 赋予了过高概率（teacher 只给了 0.1），reverse-KL 会优先压低这个错押的主峰，并把概率质量重新分配到 teacher 更支持的其他 token 上。

#### PG-style：Sample-Level Stochastic Gradient Estimator

GKD-style 的梯度需要对词表中的所有 token 求和，计算开销巨大。PG-style 的出发点很简单：能否用**单样本蒙特卡洛估计**来替代这个全词表求和？

第 4.3 节 GKD-style 的精确梯度已经给出了答案。回顾 GKD-style 的 logit 梯度，将其重写为期望形式：

$$
\nabla_\theta \text{KL}(p \| q) = \sum_{k} p_k \log \frac{p_k}{q_k} \nabla_\theta \log p_k + \underbrace{\sum_{k} p_k \nabla_\theta \log p_k}_{= \nabla_\theta \sum_k p_k = 0} = \mathbb{E}_{k \sim p} \left[ \log \frac{p_k}{q_k} \cdot \nabla_\theta \log p_k \right]
$$

右边 $\mathbb{E}_{k \sim p}[\cdot]$ 的含义是：从学生分布 $p$ 中采样一个 token $k$，计算 $\log\frac{p_k}{q_k} \cdot \nabla_\theta \log p_k$。这就是 PG-style 的梯度估计——**GKD-style 的单样本无偏估计**。

在实现上，这个采样操作自然发生：学生模型在生成过程中已经从 $p_\theta$ 采样得到了 token $a_t$，无需额外的采样步骤。PG-style 的损失函数为：

$$
\mathcal{L}_{\text{PG-OPD}}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t=1}^{T} \text{sg}(C_t) \cdot \log \pi_\theta(a_t \mid s_t) \right]
$$

其中 $C_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_T(a_t \mid s_t)$ 是单样本的 reverse-KL 近似（Schulman K1 散度近似），$\text{sg}(\cdot)$ 表示 stop-gradient。

**stop-gradient 的作用**：对上式求导时，$\text{sg}(C_t)$ 被视为常数，梯度仅通过 $\log \pi_\theta(a_t \mid s_t)$ 传播：

$$
\nabla_\theta \mathcal{L}_{\text{PG-OPD}} = \sum_t \text{sg}(C_t) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

这正是 $\mathbb{E}_{k \sim p}[\log\frac{p_k}{q_k} \cdot \nabla_\theta \log p_k]$ 的单样本实现——$C_t$ 扮演权重（"这个 token 的 student-teacher 差距有多大"），$\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 扮演更新方向（"如何调整以改变该 token 的概率"）。$C_t$ 充当了逐 token 的 cost/penalty 信号——如果 $C_t > 0$（student 在该 token 上过度押注），梯度会降低该 token 的概率；如果 $C_t < 0$（teacher 更认可该 token），梯度会提升其概率。

#### 两种实现的对比

| 维度 | GKD-style | PG-style |
|------|-----------|----------|
| **梯度涉及的 token** | 词表中所有 token | 仅采样的 1 个 token |
| **teacher 信息利用** | 完整词表分布 | 仅 sampled token 的 log-prob |
| **对未采样 token** | 精准分配（知道每个 token 该得多少） | 盲目均分（不知道 teacher 对其他 token 的态度） |
| **梯度确定性** | 确定（给定 $p$, $q$） | 随采样结果波动 |
| **方差** | 低 | 高 |
| **与 RL 融合** | 困难 | 天然兼容（$C_t$ 可叠加 ORM reward） |

两种实现实际上是计算量与稳定性的权衡：

- GKD-style 需要在每一步计算教师对**整个词表**的 logits，对于词表大小 $|\mathcal{V}| = 128\text{K}$ 的大模型，这是一笔巨大的计算开销。PG-style 只需计算教师在实际采样 token 上的 log-prob，计算量小得多。
- GKD-style 的梯度是确定性的（给定同一对分布，梯度不变），训练更稳定。而 PG-style 的梯度高度依赖采样结果——同一个前缀、同一对分布，仅仅因为采样到不同 token，梯度方向就可能截然不同，导致高方差。

另外，PG-style 的 $C_t$ 接口可以自然地叠加外部 reward 信号：

$$
\hat{A}_t = \text{sg}(C_t) + \alpha \cdot \hat{A}_{\text{ORM}}
$$

这使得 PG-style 能同时利用教师知识和任务反馈，甚至可能突破教师的能力上限。GKD-style 由于其梯度结构，难以自然地融入标量 reward。

> 在实践上，DeepSeek V4 选择了 GKD-style 的 full-vocabulary token-level 反向 KL，认为其对整个词表的精确 KL 计算提供了更稳定的梯度估计，并通过 FP4 量化教师推理、异步加载等工程手段摊销了计算开销。MiMo V2-Flash 采用了 PG-style，将教师的 log-ratio 作为 token-level 的 advantage 信号与 GRPO 的 outcome reward 结合，在计算效率和 RL 兼容性之间取得平衡。

### 4.4 OPD 与 RL 的等价性

在线蒸馏与强化学习共享相同的理论根基。PG-style OPD 的梯度直接对应 REINFORCE 策略梯度，其中教师的 log-ratio 充当逐 token 的 dense 奖励信号。更一般地，ExOPD 框架证明了 **OPD 是带 KL 约束的 RL 的一个特例**——奖励函数为 $r(x, y) = \log \pi_{\text{Teacher}}(y \mid x)$，且 KL 正则项与奖励始终等权。

这意味着 GRPO、PPO 等 on-policy RL 方法与在线蒸馏共享相同的核心哲学——在当前策略的分布上学习。区别在于信号来源：RL 使用外部奖励函数（或奖励模型），而 OPD 使用教师模型的 logits 作为密集的内在奖励。

## 5. Multi-Teacher OPD

### 5.1 从单教师到多教师

在线蒸馏消除了暴露偏差，使得学生能持续逼近单个教师的性能。但在实际应用中，一个模型通常需要同时具备多种能力——数学推理、代码生成、工具调用、长文写作——而这些能力的最优策略往往由不同的专家模型所掌握。一个在数学推理上经过充分 RL 训练的专家，未必能同时保持优秀的代码生成能力，反之亦然。

如果用单个教师在所有领域上蒸馏，学生只能学到这个教师的"综合水平"——可能在每个领域都不如该领域的最优专家。但如果轮流用不同领域的专家分别蒸馏，后一个专家的训练可能会破坏前一个专家已经教会学生的能力，即所谓的"跷跷板效应（See-Saw Effect）"。

这就引出了核心问题：**如何让学生同时从多个领域专家那里学习，同时保持各领域的峰值性能？**

### 5.2 按领域路由：每个 prompt 匹配一个教师

多教师 OPD 的核心机制是**按 prompt 的领域将不同的 prompt 路由到对应的专家教师**。给定一组领域专家教师 $\{\pi_{E_1}, \pi_{E_2}, \ldots, \pi_{E_N}\}$ 和一个路由函数 $f: \mathcal{X} \to \{1, \ldots, N\}$，多教师 OPD 的目标函数为：

$$
\mathcal{L}_{\text{MT-OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t=1}^{|y|} \text{KL}\left( \pi_\theta(\cdot \mid s_t) \| \pi_{E_{f(x)}}(\cdot \mid s_t) \right) \right]
$$

其中 $s_t = (x, a_1, \ldots, a_{t-1})$，$f(x)$ 返回 prompt $x$ 所属领域对应的教师编号。对于每个 prompt $x$，在学生采样的序列上，只计算与该领域教师 $E_{f(x)}$ 之间的反向 KL。任何时刻学生只需要面对**一个**教师——只是不同 prompt 的序列面对不同的教师。

DeepSeek V4 和 MiMo V2-Flash 都采用了这一机制。DeepSeek V4 的技术报告明确指出：训练样本在数据分发时按教师索引排序（"order training samples by teacher index during data dispatching"），每个 mini-batch 中每个教师只需加载一次，且**同一时刻设备内存中最多只存在一个教师 head**。这意味着每个训练样本只与一个教师计算 KL，而非同时面对所有教师。MiMo V2-Flash 同样使用 $\pi_{\text{domain}_x}$ 作为每个 prompt 的对应教师。

路由方案的核心优势是**计算效率**：每个 mini-batch 只需加载一个教师的 logits（或 log-prob），避免了 $N_{\text{teachers}} \times |\mathcal{V}|$ 规模的 logits 张量同时驻留内存。这对于 GKD-style（需要 full-vocabulary logits）尤为重要——词表大小 $|\mathcal{V}| = 128\text{K}$ 的场景下，同时加载多个教师的 logits 在工程上不可行。

路由函数 $f(x)$ 的实现通常是确定性的分类器——根据 prompt 的内容将其归入预定义的领域类别。在实践中，"每个教师被分配的样本比例"起到了隐式权重的作用：如果数学数据占训练集的 40%，则数学教师在整个训练过程中的影响力自然就是 40%。

### 5.3 GKD-style 与 PG-style 在多教师场景下的差异

在按域路由的多教师场景下，两种梯度实现路径的工程特性差异尤为显著。

**GKD-style（DeepSeek V4）** 使用 full-vocabulary logit 梯度。在按域路由的场景下，每个 mini-batch 只需加载路由到的那个教师的完整词表级 logits，计算开销与单教师场景相同。DeepSeek V4 通过 FP4 量化教师推理和异步权重加载来支撑超过 10 个教师模型的调度——由于路由保证同一时刻只需一个教师 head，内存压力可控。但由于 logit 梯度是词表级的向量信号，GKD-style 难以与外部标量 reward 自然组合。

**PG-style（MiMo V2-Flash）** 只需计算教师在实际采样 token 上的 log-prob。在按域路由的场景下，每步只需加载一个教师的一个 token 概率值，计算开销极低。PG-style 的独有优势在于 **advantage 的可组合性**——教师的 log-ratio 信号可以与外部 reward 信号直接线性叠加：

$$
\hat{A}_{\text{MOPD}, t} = \text{sg} \left[ \log \frac{\pi_{\text{domain}_x}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} \right] + \alpha \cdot \hat{A}_{\text{ORM}}
$$

其中 $\hat{A}_{\text{ORM}}$ 是 Outcome Reward Model（如 GRPO）提供的序列级奖励信号，$\alpha$ 是平衡两者的超参数。教师的 token-level 信号告诉学生"在每一步应该怎么走"，而 ORM 的序列级信号告诉学生"最终走到了哪里"。两者互补——dense 信号提供稳定的梯度方向，outcome 信号确保全局目标的一致性。

此外，PG-style 还可以使用 PPO 策略梯度中的许多技巧来提升探索效率：比如引入重要性采样修正分布偏移，使得旧数据可以用于更新新策略；再引入 clipping 机制 $w_t = \text{clip}\left(\frac{\pi_\theta(a_t \mid s_t)}{\mu_\theta(a_t \mid s_t)}, 1-\epsilon, 1+\epsilon\right)$ 来降低波动（关于这些技巧的原理可以参考我的另一篇[博客](https://procrastinatorrrr.github.io/posts/tech/202604-rl-llm/)）。

## 6. 总结

$$
\text{SFT（人工 + CE）} \xrightarrow{\text{数据瓶颈}} \text{教师数据 + CE} \xrightarrow{\text{one-hot 信息损失}} \\ \text{离线蒸馏（教师数据 + KL）} \xrightarrow{\text{暴露偏差、 } O(T^2)} \text{OPD（学生数据 + KL）} \\ \xrightarrow{\text{多领域融合}} \text{Multi-Teacher OPD}
$$

- **SFT** 用交叉熵在人工标注数据上训练。数据质量有限、规模有限、模型能力上限被标注者锁定。
- **教师合成数据 + 交叉熵** 用更强的模型生成训练数据，解决了数据瓶颈。但交叉熵将教师的输出压缩为 one-hot，丢弃了 token 之间的相对关系信息。
- **离线蒸馏** 用 KL 散度替代交叉熵，让学生匹配教师的完整输出分布，保留了暗知识。但训练数据仍然是教师预生成的固定数据集——训练分布不等于推理分布，暴露偏差和 $O(T^2)$ 误差累积依然存在。
- **在线蒸馏** 让学生模型在自己的分布上采样，同时由教师在学生访问的状态上提供监督。训练分布等于推理分布，暴露偏差被消除，误差降至 $O(T)$。KL 方向从 forward 翻转为 reverse——梯度从逐坐标校正变为全局重分配，可以通过 full-vocabulary exact gradient（GKD-style）或 sample-level stochastic estimator（PG-style）两种路径实现。
- **多教师 OPD** 通过加权反向 KL 之和将多个领域专家的知识融入单一学生模型，按 prompt 领域路由对应的教师，并结合 outcome reward 保证全局一致性。DeepSeek V4 和 MiMo V2-Flash 的工作表明，这一范式已经在万亿参数规模上取得了实际效果。

---

## References

- [1] Ross, S., Gordon, G., & Bagnell, D. (2011). [A reduction of imitation learning and structured prediction to no-regret online learning](https://arxiv.org/abs/1105.4603). *AISTATS 2011*.
- [2] Hinton, G., Vinyals, O., & Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). *arXiv preprint arXiv:1503.02531*.
- [3] Gu, Y., Han, X., Liu, Z., & Wang, Y. (2024). [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.04496). *ICLR 2024*.
- [4] Agarwal, A., et al. (2024). [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2402.12771). *ICLR 2024*.
- [5] LLM-Core Xiaomi (2026). [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780). *arXiv preprint arXiv:2601.02780*.
- [6] Yang, W., Liu, W., Xie, R., et al. (2026). [Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125). *arXiv preprint arXiv:2602.12125*.
- [7] DeepSeek-AI (2026). [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf).

## Citation

如果你在研究或工作中引用了本文，请以以下格式引用：

**BibTeX:**

```bibtex
@misc{long2026sft_distillation,
  author       = {Long, Yijun},
  title        = {从 SFT 到 On-Policy Distillation：训练分布与推理分布的对齐之路},
  year         = {2026},
  howpublished = {\url{https://procrastinatorrrr.github.io/posts/tech/202604-sft-to-distillation/}},
  note         = {Accessed: 2026-04-26}
}
```

**APA Style:**

```txt
Long, Y. (2026). 从 SFT 到 On-Policy Distillation：训练分布与推理分布的对齐之路. https://procrastinatorrrr.github.io/posts/tech/202604-sft-to-distillation/
```