---
title: "大模型中的强化学习：从 REINFORCE 到 xxPO"
date: 2026-03-21
author: "Yijun Long"
tags: ["RL", "LLM", "RLHF", "PPO", "GRPO", "RLVR"]
categories: ["Machine Learning", "LLM", "Reinforcement Learning"]
description: "系统梳理大模型中强化学习的核心概念与算法演进。"
math: true
summary: "系统梳理大模型中强化学习的核心概念与算法演进。" 
weight: # 输入1可以顶置文章，用来给文章展示排序，不填就默认按时间排序
slug: "202604-rl-llm"
draft: false # 是否为草稿
comments: true
showToc: true # 显示目录
TocOpen: true # 自动展开目录
autonumbering: true # 目录自动编号
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
searchHidden: false # 该页面可以被搜索到
showbreadcrumbs: true #顶部显示当前路径
mermaid: true
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

> 本文假设读者具备基本的机器学习知识，对大模型有一定了解，但不需要预先掌握强化学习的背景。

## RL Basics

> Reinforcement learning is learning what to do—how to map situations to actions——so as to maximize a numerical reward signal. ----- Richard S. Sutton and Andrew G. Barto 《Reinforcement Learning: An Introduction II》

强化学习（Reinforcement learning，RL）讨论的问题是一个智能体 (agent) 怎么在一个复杂不确定的环境 (environment) 里面去最大化它能获得的回报。通过感知所处环境的状态 (state) 对动作 (action) 的奖励 (reward) ，来指导更好的动作，从而获得最大的回报 (return) ，这样的学习方法就被称作强化学习。

其中的关键概念：

- 智能体（Agent）：学习者或决策者（如大语言模型）
- 策略（Policy, π）：智能体的行为规则，即从状态到动作的映射
- 环境（Environment）：智能体所在的世界（如文本生成环境）
- 状态（State, s）：环境在某一时刻的描述
- 动作（Action, a）：智能体可以采取的操作（如生成一个 token）
- 奖励（Reward, r）：环境对智能体动作的反馈（如奖励模型的评分）
- 轨迹（Trajectory, τ）：智能体在环境中的一次完整交互，包含状态、动作和奖励序列。
- 回报（Return）：又称cumulated future reward，即未来累积奖励。而又由于未来的奖励不如当下的奖励重要，同时为了避免无限步的任务的回报发散，会加上折扣因子 γ：
    $$
    G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + ... = \sum_{k=0}^∞ γ^k r_{t+k+1}
    $$
    强化学习的目标，就是最大化回报。

> 在LLM中，LLM本身就是Agent和Policy；环境是文本生成环境；状态是prompt+已生成的tokens；动作是下一个生成的token；奖励是对生成的文本的评分。

因为：策略 π 是随机的（在同一个状态下可能选择不同动作），环境的动态（状态转移）可能是随机的（同样的动作可能导致不同的新状态），使得每一步的奖励都是随机的，故回报是一个随机变量。故引入期望回报——即价值函数：

- 状态价值函数 $V_π(s)$：在策略 $π$ 下，从状态 $s$ 开始的期望回报：
    $$
    V_π(s) = E_{π} [G_t | s_t = s]
    $$
- 动作价值函数 $Q_π(s, a)$：在策略 $π$ 下，在状态 $s$ 下采取动作 $a$ 后的期望回报：
    $$
    Q_π(s, a) = E_{π} [G_t | s_t = s, a_t = a]
    $$

V与Q的关系：

- 从Q到V：状态价值V等于在该状态下，所有可能的动作价值Q的加权平均。
    $$
    V_{π}(s) = \sum_{a} π(a|s) Q_{π}(s, a)
    $$
- 从V到Q：动作价值Q等于执行该动作后的即时奖励，加上下一个状态的折扣后价值。
    $$
    Q_{π}(s, a) = E[r_{t+1} + γ V_{π}(s_{t+1})| s_t = s, a]
    $$

根据以上定义，有
$$
G_t = r_{t+1} + γ G_{t+1}
$$
两边取期望，得到**贝尔曼方程**：
$$
V_π(s) = E_π [r_{t+1} + γ V_π(s_{t+1})| s_t = s]
$$
$$
Q_{π}(s, a) = E_π [r_{t+1} + γ Q_{π}(s_{t+1}, a_{t+1})| s_t = s, a_t = a]
$$

> 直觉：当前价值 = 当前奖励 + 折扣 * 下一个价值

贝尔曼方程是强化学习的核心：它建立了当前与未来的递归关系，将复杂的强化学习问题分解为递归的子问题，从而使求解成为可能。从价值函数出发，我们重新定义强化学习的目标：找到最优策略 $π^*$，使得对于所有状态 $s$，$V^{π^*}(s)$ 最大。那么，如何找到最优策略 $π^*$ 呢？让我们引出最优价值函数：

- 最优状态价值函数 $V^*(s)$：所有策略中最大的状态价值，即
  $$
  V^*(s) = \max_{π} V_π(s)
  $$

- 最优动作价值函数 $Q^*(s, a)$：所有策略中最大的动作价值，即
  $$
  Q^*(s, a) = \max_{π} Q_π(s, a)
  $$

同样，讨论V\*与Q\* 的关系：

- 从 Q\*到V\*：最优状态价值等于在该状态下选择最优动作的价值：
   $$
   V^*(s) = \max_{a} Q^*(s, a)
   $$
   这是因为最优策略会在每个状态选择能带来最大价值的动作。

- 从 V\*到Q\*：最优动作价值等于执行该动作后的即时奖励，加上下一个状态的最优价值的折扣：
   $$
   Q^*(s, a) = E[r_{t+1} + γ V^*(s_{t+1}) | s_t = s, a_t = a]
   $$
   这是因为在选择动作 $a$ 后，环境会转移到新状态 $s_{t+1}$，此时最优策略会从 $s_{t+1}$ 开始继续最大化价值。

组合以上两式，得到完整的贝尔曼最优方程：
$$
V^*(s) = \max_{a} \left[ E[r_{t+1} + γ V^*(s_{t+1}) | s_t = s, a_t = a] \right]
$$
对应的动作价值形式：
$$
Q^*(s, a) = E\left[ r_{t+1} + γ \max_{a'} Q^*(s_{t+1}, a') | s_t = s, a_t = a \right]
$$
通过求解贝尔曼最优方程得到最优价值函数Q*，进而得到最优策略：
$$
π^*(a|s) = \arg\max_{a} Q^*(s, a)
$$
这便是所有强化学习算法的理论基础。

## RL Algorithms

> 本文不会囊括所有强化学习算法，仅介绍与LLM相关的算法。

### 从 REINFORCE 开始

贝尔曼最优方程告诉了我们怎么找最优策略：即总是选择Q值最大的那个动作。然而在LLM场景下，vocab size通常是几万甚至几十万，准确估计每个token的Q值极其困难，在计算上不可行。

那怎么办？最直接的想法是：既然Q难以精确计算，不如直接舍弃Q与贝尔曼方程，不通过Q函数间接得到策略，而是直接用一个参数化的函数来表示策略。设策略为 $\pi_\theta(a|s)$，其中 $\theta$ 是策略的参数（比如LLM的权重）。强化学习的目标是找到最优的 $\theta$，使得期望累积折扣奖励最大。

定义期望累积折扣回报为目标函数：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]
$$
其中轨迹 $\tau = (s_0, a_0, r_1, s_1, a_1, \dots, s_{T-1}, a_{T-1}, r_T)$ 是一个长度为 $T$ 的回合（Episode）。$G_0$ 是从时刻 $t=0$ 开始的总折扣回报：
$$
G_0 = r_1 + \gamma r_2 + \gamma^2 r_3 + \dots + \gamma^{T-1} r_T = \sum_{k=0}^{T-1} \gamma^k r_{k+1}
$$
（这里的 $G_0$ 对应于 RL Basics 部分定义的 $G_t$ 当 $t=0$ 时的形式）。

通过对 $J(\theta)$ 求梯度并沿梯度方向更新参数，就可以让策略越来越倾向于选择能带来高回报的动作。对 $J(\theta)$ 求梯度：
$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_\tau P(\tau|\theta) G_0(\tau) = \sum_\tau \nabla_\theta P(\tau|\theta) G_0(\tau)
$$
利用对数导数技巧（Log-derivative trick）的恒等式 $\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)$，得：
$$
\nabla_\theta J(\theta) = \sum_\tau P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) \cdot G_0(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau|\theta) \cdot G_0(\tau)]
$$
轨迹 $\tau$ 的生成不仅取决于策略，还取决于环境。其完整概率为：
$$
P(\tau|\theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$
两边取对数后，连乘变为连加：
$$
\log P(\tau|\theta) = \log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1}|s_t, a_t)
$$
由于环境的初始状态分布 $p(s_0)$ 和状态转移概率 $p(s_{t+1}|s_t, a_t)$ 是由客观环境决定的规律，**与策略参数 $\theta$ 无关**，因此对 $\theta$ 求梯度时，这两项的导数直接为 0。所以：
$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$
代入期望公式，并将求和符号提出：
$$
\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_0(\tau) \right]
$$

对于每一个时间步 $t$，动作 $a_t$ 及其后续动作只能影响时刻 $t+1$ 及之后的奖励。因此，任何时刻 $k \le t$ 的奖励 $r_{k+1}$ 的值，都不会受到动作 $a_t$ 及其后续动作的影响。

利用此性质，在计算策略梯度时，我们可以忽略任何发生在动作 $a_t$ 之前的奖励项。我们将 $G_0(\tau)$ 展开：$G_0(\tau) = \sum_{k'=0}^{T-1} \gamma^{k'} r_{k'+1}$。对于 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 这一项，它与 $G_0(\tau)$ 中所有 $k' < t$ 的奖励的乘积的期望为 0。因此，我们只需要考虑 $k' \ge t$ 的奖励项：
$$
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_0(\tau) \right] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \sum_{k'=t}^{T-1} \gamma^{k'} r_{k'+1} \right]
$$
为了与标准的回报 $G_t$ 定义对齐（$G_t = r_{t+1} + \gamma r_{t+2} + \dots = \sum_{j=0}^{T-t-1} \gamma^j r_{t+j+1}$），我们可以将上述和式变形：
$$
\sum_{k'=t}^{T-1} \gamma^{k'} r_{k'+1} = \gamma^t r_{t+1} + \gamma^{t+1} r_{t+2} + \dots + \gamma^{T-1} r_T = \gamma^t (r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{T-t-1} r_T) = \gamma^t G_t
$$
所以，对于每个时间步 $t$ 的期望项，可以写成：
$$
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t G_t \right]
$$
将替换后的项代回求和公式，得到策略梯度的更一般形式：
$$
\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t G_t \right]
$$
进一步，我们将上述轨迹的全局期望，分解为在状态 $s_t$ 和动作 $a_t$ 上的条件期望：
$$
\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{s_t \sim P(s_t|\theta), a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t \mathbb{E}[G_t | s_t, a_t] \right]
$$
其中，$\mathbb{E}[G_t | s_t, a_t]$ 正是从状态 $s_t$ 执行动作 $a_t$ 后的期望未来折扣回报，这正是**动作价值函数 $Q_{\pi_\theta}(s_t, a_t)$** 的定义。代入 $Q$ 值后，式子变为：
$$
\nabla_\theta J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{s_t, a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t Q_{\pi_\theta}(s_t, a_t) \right]
$$
最后，利用**期望的线性性质（期望的和等于和的期望）**，我们将外侧的求和符号 $\sum_{t=0}^{T-1}$ 移到期望的内部。当所有时间步被加总在一起时，对单个 $(s_t, a_t)$ 的期望就自然等价于对一整条轨迹 $\tau$ 的期望。由此，我们得到了通用的**策略梯度定理（Policy Gradient Theorem）**：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t Q_{\pi_\theta}(s_t, a_t) \right]
$$

定理非常优美，但回到现实，我们依然无法精确计算 $Q_{\pi_\theta}(s_t, a_t)$。不过，由于 $Q_{\pi_\theta}(s_t, a_t)$ 的定义就是 $G_t$ 的数学期望，这意味着**在环境中单次实际采样得到的 $G_t$，天生就是 $Q_{\pi_\theta}(s_t, a_t)$ 的无偏估计（Unbiased Estimator）**。

因此，**蒙特卡洛方法（Monte Carlo）** 提出了一种粗暴但有效的近似：直接用单次采样得到的真实回报 $G_t$，去估计那个算不出来的理论期望 $Q_{\pi_\theta}(s_t, a_t)$。由此我们得到了可以用于代码计算的 **REINFORCE 经验梯度**：
$$
\nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t G_t
$$
利用这个采样出来的梯度，我们可以通过梯度上升来优化策略函数，即：
$$
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t G_t
$$
其中 $\alpha$ 是学习率。这就是 **REINFORCE** 算法的核心。其伪代码如下：

**初始化**：策略网络 $\pi_\theta$，学习率 $\alpha$，折扣因子 $\gamma$  
for every episode（轨迹）：

1. **生成轨迹** $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T)$  
   根据当前策略 $\pi_\theta$ 与环境交互，记录每一步的 $(s_t, a_t, r_{t+1})$ 和 $\log \pi_\theta(a_t|s_t)$
2. **计算回报** $G_t$  
   对于 $t = 0, 1, \dots, T-1$：
   $$ G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1} $$
   > 实践中常用反向迭代高效计算：由 $G_{T-1} = r_T$ 开始，反向迭代计算 $G_{T-2} = r_{T-1} + \gamma G_{T-1}$，$\dots$，$G_0 = r_1 + \gamma G_1$。
3. **计算损失并更新策略**  
   损失函数（PyTorch风格，使用梯度下降最小化负目标）：
   $$ \mathcal{L}(\theta) = -\sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) \cdot \gamma^t G_t $$
   $$ \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta) $$
4. **清空轨迹缓存**，继续下一轮循环

**说明**：

- 这是一个蒙特卡洛方法：必须等一个完整的 episode 结束后才能更新；
- 损失函数中的负号是因为我们想最大化期望回报，而优化器通常做梯度下降。
- 在现代深度强化学习的工程实践中，为了避免回合后期状态的梯度消失，通常会丢弃外层的 $\gamma^t$ 衰减，平等地对待每一个时间步，这被证明能带来更好的性能。此时，策略梯度变为：
  $$
  \nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t
  $$

### REINFORCE 的发展

REINFORCE是最基础的策略梯度方法，它有个很严重的问题：它直接使用蒙特卡洛采样的回报 $G_t$ 来评价动作，而 $G_t$ 是从当前策略采样一整条轨迹得到的，包含了环境动态和策略随机性的累积影响，导致不同轨迹之间的 $G_t$ 波动非常大，从而使得梯度更新方向不稳定，收敛缓慢。如何减小波动呢？

> 直觉：我们希望只关注当前动作带来的增益，剥离环境带来的影响。

因此，将 $G_t$ 减去一个只依赖于环境s，不依赖动作a的baseline $b(s)$，使策略梯度变为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t-b(s_t)) \right]
$$

由于 $b(s)$ 与a无关，有

$$
\mathbb{E}_{\tau \sim \pi_\theta} \left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \right] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0
$$

因此，减去baseline的技巧并不会影响策略梯度的计算结果（无偏性），但是可以降低波动。那如何取baseline呢？

> 直觉：好的baseline应该准确反映环境带来的增益——即 $b(s) = \mathbb{E}_\pi[G_t|s_t=s] = V_\pi(s)$。此时，$G_t - b(s)$ 就是动作带来的增益。

可以证明，$b(s) = V_\pi(s)$ 就是最优的baseline（能够最大程度降低波动）。在此时，$G_t - b(s)$ 的期望为：

$$
\mathbb{E}_\pi \left[ G_t-b(s_t) | s_t=s, a_t=a \right] = Q_\pi(s, a) - V_\pi(s)
$$

由此，我们引入**优势函数（Advantage Function）**：

$$
A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)
$$

优势函数的意义：反映了动作a比“平均动作“好多少。使用优势函数后，策略梯度定理变为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_\pi(s_t, a_t) \right]
$$

到这里一切看起来都很自然。但是等等——我们一开始引入策略梯度，不就是因为Q无法计算吗？但是此处为了降低波动引入优势函数 $A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$，却又重新引入了Q与V。因此，我们需要对优势函数进行估计。假设我们能够使用 $V(s)$ 估计 $V_\pi(s)$：

- 第一种估计方法：我们知道， $G_t$ 是 $Q_\pi(s, a)$ 的无偏估计。则，$A_\pi(s, a)$ 可以估计为：

  $$
  \hat{A}_t = G_t - V(s)
  $$

  我们称此估计方法为蒙特卡洛（Monte Carlo，MC）方法。MC方法的好处是：由于$G_t$ 是 $Q_\pi(s, a)$ 的无偏估计，因此MC估计不会引入任何的额外偏差，所有的偏差来自 $V(s)$ 的估计。而代价依然是波动：因为 $G_t$ 依赖于未来所有步骤的随机性（动作采样、状态转移、奖励噪声），轨迹越长，方差累积越大。

- 第二种估计方法：受 $Q_{π}(s, a) = E[r_{t+1} + γ V_{π}(s_{t+1})| s_t = s, a]$ 启发，我们可以估计 $A_\pi(s, a)$ 为：

  $$
  \hat{A}_t = r_{t+1} + γ V(s_{t+1}) - V(s)
  $$

  我们称此估计方法为时序差分（Temporal Difference，TD）方法，将 $δ_t = r_{t+1} + γV(s_{t+1}) - V(s_t)$ 称为单步TD误差。TD方法最大的好处是降低波动：因为TD只依赖单步奖励 $r_t$ 和下一个状态的估计，不累积未来多步的随机性，因此方差显著降低；而代价是，由于 $V(s)$ 存在估计误差，而TD会将此误差放大。

- 第三种估计方法：MC方法方差大但无偏，TD方法方差小但有偏。能否结合两者优势？我们定义n步回报：

  $$
  G_t^{(n)} = r_{t+1} + γ r_{t+2} + ... + γ^{n-1} r_{t+n} + γ^n V(s_{t+n})
  $$

  然后定义n步TD估计为：
  $$
  \hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)
  $$

  当n=1时，退化为单步TD；当n=∞时，退化为MC。那如何选取最优的n呢？与其手动选择，不如对所有的n步回报进行加权平均，这就是**广义优势估计（Generalized Advantage Estimation，GAE）**：

  $$
  \hat{A}_t^{GAE(γ, λ)} = \sum_{l=0}^{∞} (γλ)^l δ_{t+l}
  $$

  其中 $δ_t = r_{t+1} + γV(s_{t+1}) - V(s_t)$ 是TD误差，$λ ∈ [0, 1]$ 是权衡参数。

  GAE的直观理解：
  - 当 $λ = 0$ 时，$\hat{A}_t = δ_t$，退化为单步TD，偏差大、方差小；
  - 当 $λ = 1$ 时，$\hat{A}_t = \sum_{l=0}^{∞} γ^l δ_{t+l} = G_t - V(s_t)$，退化为MC，无偏但方差大；
  - $λ$ 在0和1之间时，在偏差和方差之间取得平衡。

  GAE的优势在于：通过调节超参数λ，我们可以灵活地控制偏差-方差的权衡，而无需手动选择步数n。这使得GAE成为PPO等算法的默认优势估计方法。在实际应用中，$\lambda$ 通常默认取0.95。

### Actor-Critic

注意到：上文的各种优势估计方法，都有一个默认的前提：我们能够使用 $V(s)$ 估计 $V_\pi(s)$。然而，又该如何让 $V(s)$ 能够较为准确的估计 $V_\pi(s)$ ？显然，$V$ 不能随意选取，想得到它同样需要进行训练。那是否可以提前训练好 $V$？答案是不行——因为 $V$ 是对 $V_\pi(s)$ 的估计，而 $V_\pi(s)$ 是随着策略的不断更新在动态变化的，因此 $V$ 同样应该是随着策略更新而动态变化的。所以，必须同时训练策略 $\pi$ 与价值 $V$。这便是**Actor-Critic**架构的核心。具体来说，Actor-Critic架构同时训练以下两个网络：

- Actor（策略网络）$\pi_\theta(a|s)$：负责生成动作；
- Critic（价值网络）$V_\phi(s)$：负责估计状态价值 $V_\pi(s)$。

Critic以价值估计的均方误差为损失函数，使用梯度下降更新参数；而Actor利用Critic给出的评分（即 $V(s)$）计算优势函数（使用上文提到的估计方法），然后使用策略梯度更新参数。

Actor-Critic结合了策略梯度和价值函数的优点：

- 相比纯策略梯度（REINFORCE）：Critic提供了更稳定、方差更小的梯度估计；
- 相比纯价值方法（如DQN）：Actor直接输出动作概率，适用于连续动作空间和大规模离散动作空间（如LLM的token生成）。

### Importance Sampling

Actor-Critic 成功解决了REINFORCE算法的高方差问题，但REINFORCE算法仍然存在其他问题需要解决。标准的策略梯度定理告诉我们，为了优化策略 $\pi_\theta$，我们需要计算以下期望：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_\pi(s_t, a_t) \right]
$$

该期望 $\mathbb{E}_{\tau \sim \pi_\theta}$ 是基于当前策略 $\pi_\theta$ 产生的数据分布。这意味着：你只能用当前最新策略去环境里采样，算一次梯度，更新一次参数。一旦参数 $\theta$ 更新变成了 $\theta'$，旧数据（由 $\theta$ 产生）就失效了，不能再次用于训练 $\theta'$。这意味着每次更新都要重新与环境交互，样本效率极低。

为了提高效率，我们希望能利用旧策略 $\pi_{\theta_{old}}$ 收集的数据来更新新策略 $\pi_\theta$。然而，如果直接使用旧数据更新新策略，会出现以下问题：我们想要计算的梯度期望是 $\mathbb{E}_{\pi_\theta} [ \dots ]$，但我们手头有的数据采样自 $\pi_{\theta_{old}}$。如果我们直接用旧数据计算新策略的梯度，由于分布偏移（Distribution Shift），估计将是有偏的（Biased）。

为了解决上述的分布不一致问题，我们需要一种数学方法，将“基于 $\pi_\theta$ 的期望”转换为“基于 $\pi_{\theta_{old}}$ 的期望”。这就是重要性采样的恒等式：

$$ \mathbb{E}_{x \sim P}[f(x)] = \mathbb{E}_{x \sim Q} \left[ \frac{P(x)}{Q(x)} \cdot f(x) \right] $$

将其应用到策略梯度中：

- $P(x) = \pi_\theta(a|s)$ (目标分布)
- $Q(x) = \pi_{\theta_{old}}(a|s)$ (采样分布)
- $f(x) = \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)$

得到：
$$
\nabla_\theta J(\theta) \approx \mathbb{E}_{\tau \sim \pi_{\theta_{old}}} \left[ \sum_{t=0}^{T} \underbrace{\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}}_{\text{重要性采样比率 } \rho_t} \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t) \right]
$$

通过引入比率 $\rho = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$，我们成功修正了分布差异，使得旧数据可以用于更新新策略。

### TRPO（Trust Region Policy Optimization）

重要性采样通过引入比率 $\rho$，成功修正了分布差异，使得旧数据可以用于更新新策略，从而成功解决了REINFORCE算法样本效率低的问题。但是，凡事都有代价：重要性采样又重新带回了本来通过AC架构已经赶走的“波动过大（高方差）”的问题。具体来说，如果 $\pi_{\theta_{old}}(a|s)$ 非常小（旧策略觉得这个动作几乎不可能），而 $\pi_\theta(a|s)$ 较大，那么比率 $\rho$ 会变得非常大。这会导致梯度剧烈波动，进而导致训练不稳定甚至崩溃。

为了解决 IS 带来的训练不稳定问题，TRPO 的作者（John Schulman 等）在 2015 年提出了一个核心理论问题：“我们能否保证每一次策略更新，性能都至少不会下降？”

针对这个问题，该论文基于保守策略迭代 (Conservative Policy Iteration) 理论推导出了一个下界。结论是：如果限制新策略与旧策略之间的 KL 散度 (Kullback-Leibler Divergence) 不超过某个阈值 $\delta$，那么策略的性能提升是有理论保证的。根据此结论，论文提出了信任域策略优化（Trust Region Policy Optimization，TRPO）——对目标函数 IS 加上 KL 散度作为硬约束，得到以下优化目标：

$$
\begin{aligned}
& \underset{\theta}{\text{maximize}}
& & \mathbb{E}_{s \sim \pi_{old}, a \sim \pi_{old}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s, a) \right] \\
& \text{subject to}
& & \mathbb{E}_{s \sim \pi_{old}} \left[ KL(\pi_{\theta_{old}}(\cdot|s) \ || \ \pi_\theta(\cdot|s)) \right] \le \delta
\end{aligned}
$$

其中：

- **目标函数 (Maximize)：** 依然使用重要性采样比率 $\rho$ 加权优势函数。这意味着我们仍然希望利用旧数据来提升回报。
- **约束条件 (Subject to)：** 新旧策略的平均 KL 散度不能超过 $\delta$（例如 0.01），这强制要求新策略不能“背叛”旧策略太远。

### PPO（Proximal Policy Optimization）

TRPO从理论上解决了IS训练不稳定的问题，理论很优美；但代价是，它的优化目标相比IS引入了不等式约束，而标准的随机梯度下降 (SGD/Adam) 很难处理这种不等式约束。为了求解这个约束，TRPO 不得不使用复杂的二阶优化（费雪矩阵、共轭梯度），这导致了各种各样的问题：如计算量过大、工程实现过于复杂、兼容性差、难以进行分布式训练等等……

能否找到一种方法，不需要二阶优化，不需要复杂约束求解，却能像 TRPO 一样限制策略更新幅度？

#### PPO-Penalty

一种想法来自拉格朗日松弛 (Lagrangian Relaxation)：将“带有硬约束的优化问题”，转化为“带有惩罚项的无约束优化问题”。这便是 **PPO-Penalty (Proximal Policy Optimization with KL Penalty)** 算法。具体来说，我们把 TRPO 的约束条件 $KL(\pi_{\theta_{old}}(\cdot|s) \ || \ \pi_\theta(\cdot|s)) \le \delta$ 看成一个惩罚项，直接加入目标函数中，得到如下目标函数：

$$
J(\theta) = \underbrace{\mathbb{E}_t \left[ r_t(\theta) \hat{A}_t \right]}_{\text{IS 目标函数}} - \underbrace{\beta \cdot KL(\pi_{\theta_{old}}(\cdot|s_t) \ || \ \pi_\theta(\cdot|s_t))}_{\text{KL 散度惩罚项}}
$$

通过在目标函数中引入 KL 散度惩罚项，我们将TRPO中的硬约束变成了软约束：为了最大化目标函数，策略会自动控制新旧策略之间的偏离幅度，因为当 $\pi_\theta$ 偏离 $\pi_{\theta_{old}}$ 太远时，$KL$ 散度项会迅速增大，从而抵消掉通过优势函数 $\hat{A}_t$ 获得的增益。这样一来，算法既达成了 TRPO 的约束目标，又避免了二阶优化的复杂性。

在实际应用中，固定的惩罚系数 $\beta$ 不容易确定：如果 $\beta$ 太小，约束力不足，策略更新依然可能由于步长过大而崩溃；如果 $\beta$ 太大，策略更新会变得过于保守，导致收敛极其缓慢。因此，需要对 $\beta$ 进行动态调整，才能使得算法在不同的训练阶段都能维持一个相对稳定的更新幅度。

#### PPO-Clip

重新分析 IS 带来的高方差问题：如果 $\pi_{\theta_{old}}(a|s)$ 非常小（旧策略觉得这个动作几乎不可能），而 $\pi_\theta(a|s)$ 较大，那么比率 $\rho$ 就会变得非常大，导致梯度剧烈波动，进而导致训练不稳定甚至崩溃。

> 直觉：既然是比率 $\rho$ 过大导致的梯度波动，为何不直接限制 $\rho$ 的范围？

这就是 PPO-Clip 的核心思想：与其在目标函数后面加惩罚项，不如直接对**重要性采样比率 $r_t(\theta)$** 进行“截断”。其目标函数如下：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

其中：

- **$\text{clip}$ 函数**：将 $r_t(\theta)$ 限制在 $[1-\epsilon, 1+\epsilon]$ 之间（通常 $\epsilon=0.2$）。
- **$\min$ 操作**：取“原始目标”和“裁剪目标”中的最小值。这确保了当新策略带来的好处（优势项）过大时，我们依然强制限制其更新幅度，不让策略在单次更新中发生突变。

通过 clip，PPO-Clip 直截了当的解决了 IS 高方差的问题。它形式非常简单，但是大量实验却证明其效果非常好。因此，PPO-Clip 成为了目前大语言模型（LLM）对齐（RLHF）阶段的主流算法。

## RL in LLM

至此，LLM 时代前的 RL 算法发展基本介绍完成。接下来，将结合具体的 LLM RL 场景，介绍 当前主流的 RL 算法。

在讨论具体场景之前，需要先回答一个问题：问什么 LLM 训练需要 RL？

我们都知道如今 LLM 训练的三部曲：大规模预训练 - SFT - RL。其中，无论是预训练还是 SFT，使用的损失函数均为交叉熵损失。在最小化损失函数的训练过程中，模型实际上在逐token地模仿提供给它的训练数据。但是，这样的策略在实际应用中，有着许多的局限：

- 模型在训练时，是在给定完美上文的情况下预测下一个词；然而在生成时，模型是自回归地进行生成，一旦模型在某一个 token 上出错，接下来的所有 token 都会被带偏，而交叉熵损失无法缓解这个问题；
- 单纯的模仿很难让模型与人类的价值观对齐：模型难以知道什么样的回答是人类喜欢的，什么样的回答是有帮助的，而什么样的回答是有害的；
- 对于各种复杂的推理问题、数学问题、代码问题，单纯让模型模仿人类的解题过程很难让模型真正理解其背后的逻辑，导致模型的输出往往流于表面形式。

引入 RL，正是为了解决以上各种问题。首先尝试将 RL 应用到 LLM 中用于对齐的工作来自 OpenAI 的 [InstructGPT](https://arxiv.org/abs/2203.02155v1)（ChatGPT的前身）。

### InstructGPT 与 RLHF 范式

InstructGPT 确立了如今大语言模型对齐的标准范式——**基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）**。它的核心思想是：既然我们无法用一个简单的数学公式来定义什么是“好”的回答，那我们就先让人类来打分，训练一个“裁判模型”来模拟人类的偏好，然后再让 LLM 作为一个智能体（Agent）在这个裁判的反馈下进行强化学习。

InstructGPT 的训练流程包含三个标准步骤（即著名的“三步曲”）：

- 第一步：监督微调（Supervised Fine-Tuning, SFT）

  在这个阶段，我们收集少量且高质量的人类示范数据（Prompt + 期望的完美回答），利用传统的交叉熵损失函数（Cross-Entropy）对预训练大模型进行微调。
  这一步的目的是**让模型学会“人类对话的格式”和“基本遵循指令的能力”**。经过这一步训练得到的模型被称为 $\pi^{\text{SFT}}$。虽然 $\pi^{\text{SFT}}$ 已经能正常对话了，但正如前文所述，它受限于模仿学习的本质，存在过度迎合、容易产生幻觉等问题。

- 第二步：训练奖励模型（Reward Model, RM）

  既然 RL 需要一个奖励信号（Reward），而在文本生成任务中没有现成的环境分数，我们就自己训练一个。

  1. **数据收集**：给定一个 Prompt $x$，我们让 SFT 模型 $\pi^{\text{SFT}}$ 生成多个不同的回答（比如 $y_1, y_2, y_3, y_4$）。然后，让人类标注员对这几个回答的质量进行**排序（Ranking）**，而不是直接打分（因为直接打分主观性太强，而比较好坏相对容易且一致性高）。假设人类认为 $y_w$ 比 $y_l$ 好（$y_w \succ y_l$）。
  2. **模型训练**：我们拿出一个与 SFT 模型结构相似的模型（通常砍掉最后的词表预测层，换成一个标量输出层），它的输入是 Prompt + 回答 $(x, y)$，输出是一个代表分数的标量 $r_\phi(x, y)$，其中 $\phi$ 是 RM 的参数。

  RM 的训练目标是：**让好回答的得分尽量高于坏回答的得分**。我们使用 Bradley-Terry 模型来将排序转化为概率，最小化以下**成对排序损失（Pairwise Ranking Loss）**：

  $$
  \mathcal{L}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
  $$

  其中，$\sigma$ 是 Sigmoid 函数。这个公式非常直观：当好的回答得分 $r_\phi(x, y_w)$ 远大于坏的回答得分 $r_\phi(x, y_l)$ 时，差值为较大的正数，$\sigma$ 接近 1，$\log(1)=0$，损失极小；反之则损失很大。

- 第三步：使用 PPO 进行强化学习优化（RLHF）

  现在我们有了可以对话的初始策略 $\pi^{\text{SFT}}$，也有了代替人类打分的裁判 $r_\phi$，接下来就可以开始使用 PPO-Clip 进行强化学习了。

  在 LLM 的 PPO 训练中，我们要防止一个致命问题：**奖励黑客（Reward Hacking）**。如果让 LLM 毫无顾忌地去最大化 RM 的得分，它很快就会发现 RM 的漏洞，输出一些毫无意义但包含“高分特征词”的乱码（比如满屏的 "Thank you", "Helpful"）。
  为了解决这个问题，InstructGPT 在奖励函数中引入了**KL 散度惩罚项**。

  对于一个 Prompt $x$ 和生成的回答 $y$，我们定义经过修改的奖励函数 $R(x, y)$：

  $$
  R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}
  $$

  > **直觉**：
  > 1. $r_\phi(x, y)$ 是裁判给出的“人类偏好分”。
  > 2. $\pi_\theta^{\text{RL}}$ 是当前正在通过 RL 训练的模型（Actor），$\pi^{\text{SFT}}$ 是冻结参数的初始微调模型（Reference Model）。
  > 3. $-\beta \log \frac{\pi_\theta^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}$ 是 KL 惩罚项。它的意思是：当前模型 $\pi_\theta^{\text{RL}}$ 的输出分布不能偏离最初学会说人话的 $\pi^{\text{SFT}}$ 太远。如果偏离太远（即使 RM 给分高），就会受到扣分惩罚。这保证了模型在迎合人类偏好的同时，依然能保持语言的流畅性和基本的常识。

  将上述奖励代入我们前文推导过的 PPO-Clip 算法中，InstructGPT 最终的 RL 目标函数（去掉了加入预训练数据计算的 PPO-ptx 细节以突出核心）为最大化以下期望：

  $$
  \text{Objective}(\theta) = \mathbb{E}_{x \sim P(x), y \sim \pi_{\theta_{old}}(\cdot|x)} \left[ \frac{1}{|y|}\sum_{t=1}^{|y|} \min \left( \rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]
  $$

  其中，优势函数 $\hat{A}_t$ 是由 Critic 网络（价值函数 $V_\psi$）利用修改后的带有 KL 惩罚的奖励 $R(x, y)$ 通过 GAE（广义优势估计）计算出来的；$\rho_t(\theta) = \frac{\pi_\theta^{\text{RL}}(a_t|s_t)}{\pi_{\theta_{old}}^{\text{RL}}(a_t|s_t)}$ 是 PPO 中的重要性采样比率；而除以序列长度 $|y|$ 是为了剥离长度因素，消除“序列长度”对梯度幅度的干扰。

在实际工程落地时，执行上述算法意味着我们要同时在显存中存在 **4 个大模型**：

1. **Actor Model ($\pi_\theta$)**：正在训练的策略模型（初始化为 SFT 模型）。
2. **Reference Model ($\pi_{\text{ref}}$)**：冻结权重的参考模型（即原始的 SFT 模型，用于计算 KL 惩罚）。
3. **Reward Model ($r_\phi$)**：冻结权重的奖励模型（提供外部 Reward 分数）。
4. **Critic Model ($V_\psi$)**：正在训练的价值模型（通常初始化为 Reward Model，把最后一层改为输出每个 token 的价值预测 V 值，用于计算 Advantage）。

**完整的 PPO in LLM 训练伪代码：**

```python
# 初始化
Actor = load_model(SFT_weights)          # 待优化的策略网络 \pi_\theta
Critic = load_model(RM_weights_as_base)  # 待优化的价值网络 V_\psi
Ref_Model = load_model(SFT_weights)      # 冻结，\pi^{SFT}
Reward_Model = load_model(RM_weights)    # 冻结，r_\phi

Ref_Model.eval()
Reward_Model.eval()

beta = 0.1         # KL 散度惩罚系数
epsilon = 0.2      # PPO Clip 阈值
learning_rate = 1e-5

for batch_prompts in Dataloader:
    
    # ---------------- 1. 经验采集阶段 (Rollout / Experience Gathering) ----------------
    # Actor 产生回答并记录轨迹
    with torch.no_grad():
        responses = Actor.generate(batch_prompts)  
        
        # 获取当前 Actor 与 Ref Model 在每一步的对数概率
        logprobs_actor = Actor.get_logprobs(batch_prompts, responses)
        logprobs_ref = Ref_Model.get_logprobs(batch_prompts, responses)
        
        # 获取 Critic 对每个 token 状态的价值预测
        values = Critic(batch_prompts, responses)  
        
        # 获取 RM 对整个句子的最终评分
        rm_scores = Reward_Model(batch_prompts, responses) 

    # ---------------- 2. 奖励计算与 GAE 优势估计 (Reward & Advantage) ----------------
    rewards = []
    # 遍历生成的每一个 token 步骤 (t = 1 to T)
    for t in range(T):
        # 计算在 token t 的 KL 惩罚作为单步即时奖励（均为负值）
        kl_penalty = - beta * (logprobs_actor[t] - logprobs_ref[t]) 
        
        if t == T - 1:
            # 只有在序列最后一个 token 处，加上 RM 给出的完整句子评分
            step_reward = rm_scores + kl_penalty
        else:
            step_reward = kl_penalty
            
        rewards.append(step_reward)
    
    # 使用 GAE 计算优势函数 (Advantages) 和 回报 (Returns)
    advantages = compute_GAE(rewards, values, gamma=0.99, lambda=0.95)
    returns = advantages + values
    
    # ---------------- 3. PPO 模型更新阶段 (Optimization) ----------------
    # 将采集到的数据打乱，进行 K 个 epoch 的网络更新
    for _ in range(K_epochs):
        # 重新前向传播，计算当前最新 Actor 参数下的概率分布和 Critic 的新 V 值
        new_logprobs_actor = Actor.get_logprobs(batch_prompts, responses)
        new_values = Critic(batch_prompts, responses)
        
        # PPO-Clip 计算
        ratio = torch.exp(new_logprobs_actor - logprobs_actor.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        
        # Actor 损失（最大化目标等价于最小化负号）
        actor_loss = - torch.min(surr1, surr2).mean()
        
        # Critic 损失（均方误差）
        critic_loss = MSE(new_values, returns)
        
        # 总损失
        total_loss = actor_loss + c1 * critic_loss
        
        # 反向传播与参数更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

通过训练一个与人类偏好相同的奖励模型进行打分，然后使用该分数作为奖励进行强化学习，这种策略同时解决了我们前文提到的前两个问题：

- 对于逐误差token累积的问题，由于 RM 给出的奖励是序列级的（通过 GAE 将奖励分配到每个 token），因此 RLHF 优化的是“整条回复的质量”，而不是单个词的准确率。这让模型学会如何规划整段回答的结构，而不是目光短浅地只顾着下一个词。
- 对于对齐人类偏好的问题，RLHF 通过引入 RM，把人类的主观偏好转化为一个标量分数。RL 算法通过不断最大化这个分数，让模型自动去摸索如何让回答更符合人类的胃口。它将难以数学化的偏好问题，转化为了一个可以求解的最优化问题。

### DPO

InstrcutGPT 提出的 RLHF 范式中，使用的 RL 算法是 PPO-Clip。PPO 算法有着低方差、训练稳定的优势（前文已经分析过），这也是 OpenAI 选择它的主要原因。然而，我们也提到，在 PPO 训练过程中，我们要在显存中同时存放策略模型、价值模型、奖励模型、参考模型这4个 LLM，其中策略模型与价值模型还要同时进行训练，同时奖励模型还需要另外训练。这带来了巨大的时间开销与计算开销。同时，由于需要同时训练两个模型，PPO 的收敛速度往往较慢，这又加剧了开销。因此，降低开销就成了接下来许多算法的出发点。

其中，最激进的想法来自斯坦福大学发表的论文 DPO（[*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290v3)）。该论文提出：我们根本不需要单独训练一个奖励模型（Reward Model），也不需要使用复杂的强化学习算法（如 PPO）来进行对齐，因为**大语言模型本身就可以作为一个“隐藏的”奖励模型**。

通过严格的数学推导，DPO 将传统的 RLHF 中“拟合奖励模型 + 强化学习最大化奖励”这两个分离的步骤，完美地等价替换为了一个单一的**直接在偏好数据上进行分类的损失函数**。

- DPO 的数学推导

  我们先回到 RLHF 阶段，带 KL 散度惩罚的强化学习目标函数可以写为（对于给定的 prompt $x$）：

  $$
  \max_{\pi} \mathbb{E}_{y \sim \pi(\cdot|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
  $$

  即，我们要优化的目标是 $J(\pi) = \sum_y \pi(y|x) \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$。
  注意，这里有一个隐藏的硬约束条件：**$\pi$ 必须是一个合法的概率分布**。即对于任意给定的 $x$，都有：
  $$
  \sum_y \pi(y|x) - 1 = 0
  $$

  这是一个带约束的极值问题，我们使用**拉格朗日乘数法**，引入乘数 $\lambda$，构造拉格朗日函数 $\mathcal{L}$：
  $$
  \mathcal{L}(\pi, \lambda) = \sum_y \pi(y|x) \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right] + \lambda \left( 1 - \sum_y \pi(y|x) \right)
  $$

  为了找到极值点，我们对连续变量 $\pi(y|x)$ 求偏导数并令其等于 0。

  $$
  \frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x, y) - \beta \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + 1 \right) - \lambda = 0
  $$

  接下来我们解这个方程，把 $\pi(y|x)$ 提出来：
  $$
  \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} = r(x, y) - \beta - \lambda
  $$
  $$
  \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{r(x, y)}{\beta} - \frac{\beta + \lambda}{\beta}
  $$
  两边同时取指数 $\exp$：
  $$
  \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} = \exp\left( \frac{r(x, y)}{\beta} \right) \exp\left( - \frac{\beta + \lambda}{\beta} \right)
  $$
  $$
  \pi(y|x) = \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right) \exp\left( - \frac{\beta + \lambda}{\beta} \right)
  $$

  此时，式子里还有一个未知的 $\lambda$。我们如何消掉它？
  再次利用我们的约束条件 $\sum_y \pi(y|x) = 1$，对上式两边同时对 $y$ 求和：
  $$
  \sum_y \pi(y|x) = \sum_y \left[ \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right) \exp\left( - \frac{\beta + \lambda}{\beta} \right) \right] = 1
  $$
  由于 $\exp\left( - \frac{\beta + \lambda}{\beta} \right)$ 中不包含 $y$，它可以作为一个常数提出来：
  $$
  \exp\left( - \frac{\beta + \lambda}{\beta} \right) \sum_y \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right) = 1
  $$
  由此解得这个常数项的值：
  $$
  \exp\left( - \frac{\beta + \lambda}{\beta} \right) = \frac{1}{\sum_y \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right)}
  $$
  令 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$，代回 $\pi(y|x)$ 的表达式中，就得到了最终的最优解：

  $$
  \pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
  $$

  我们称 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 是一个配分函数（Partition Function），用来保证所有的概率加起来等于 1。

  > **直觉**：最优策略 $\pi^*(y|x)$ 等于原始的参考策略 $\pi_{\text{ref}}(y|x)$ 乘上一个由奖励 $r(x,y)$ 决定的放大/缩小系数。如果 $r$ 很大，这个回答生成的概率就会被放大；如果 $\beta$ 很大（KL惩罚很重），放大的幅度就会受到抑制。

  在传统的 RLHF 中，我们是用 PPO 去一步步逼近这个 $\pi^*$。但 DPO 的作者反向思考：既然我们有了最优策略 $\pi^*$ 和奖励 $r$ 之间的解析关系，我们为什么不**把 $r$ 反向表示出来**呢？

  对上式两边取对数并移项，我们可以用最优策略 $\pi^*$ 来表达隐藏的奖励函数 $r(x,y)$：

  $$
  r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
  $$

  让我们回想一下，在 RLHF 第二步（训练奖励模型）时，我们使用的是基于 Bradley-Terry 模型的排序损失：

  $$
  P(y_w \succ y_l | x) = \sigma \left( r(x, y_w) - r(x, y_l) \right)
  $$

  把刚才推导出的 $r(x,y)$ 的等式，代入到 $r(x, y_w) - r(x, y_l)$ 中：

  $$
  r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta \log Z(x) \right)
  $$

  注意到配分函数 $Z(x)$ 相减抵消了，剩下的是：

  $$
  r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
  $$

  我们将这个差值代回 Bradley-Terry 模型的负对数似然损失中，就得到了 **DPO 的最终损失函数**：

  $$
  \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
  $$

  > **直觉**：这个损失函数在做什么？
  > 设 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，我们可以把它看作是策略模型隐式定义的“奖励”。
  > DPO 损失要求 $\hat{r}_\theta(x, y_w)$ 尽量大于 $\hat{r}_\theta(x, y_l)$。
  > 具体来说，梯度会去**增加**被偏好回答 $y_w$ 的生成概率，同时**降低**被拒绝回答 $y_l$ 的生成概率。并且由于 Sigmoid 函数的特性，如果模型已经对这两个回答的概率排对了序且差值足够大，梯度就会变小，防止过度优化。

使用了 DPO 后，原本 InstructGPT 复杂的“三步曲”被大幅简化为了两步：

1. **第一步：监督微调（SFT）**
  与 RLHF 一样，使用高质量数据训练出 $\pi^{\text{SFT}}$。
1. **第二步：直接偏好优化（DPO）**
  收集人类偏好数据 $(x, y_w, y_l)$。直接把 $\pi^{\text{SFT}}$ 当作参考模型 $\pi_{\text{ref}}$ 冻结起来；然后再复制一份 $\pi^{\text{SFT}}$ 作为策略模型 $\pi_\theta$。直接使用上述的 DPO 损失函数，像常规的监督学习那样去更新 $\pi_\theta$ 的参数。

**DPO in LLM 伪代码实现：**

```python
# 初始化
Policy_Model = load_model(SFT_weights)  # 待优化的策略网络 \pi_\theta
Ref_Model = load_model(SFT_weights)     # 冻结的参考网络 \pi_{ref}

Ref_Model.eval()

beta = 0.1         # 控制对参考模型 KL 偏离程度的超参数
learning_rate = 1e-6

# 注意：DPO 不需要与环境交互（Rollout），直接读取离线的偏好数据集即可
for batch in Preference_Dataloader:
    # batch 包含: 
    # prompts: x
    # chosen_responses: y_w
    # rejected_responses: y_l
    
    # ---------------- 1. 前向传播，计算 Log Probs ----------------
    with torch.no_grad():
        # 获取冻结参考模型对 Chosen 和 Rejected 的对数概率
        ref_logprobs_w = Ref_Model.get_seq_logprobs(prompts, chosen_responses)
        ref_logprobs_l = Ref_Model.get_seq_logprobs(prompts, rejected_responses)
        
    # 获取当前策略模型对 Chosen 和 Rejected 的对数概率
    policy_logprobs_w = Policy_Model.get_seq_logprobs(prompts, chosen_responses)
    policy_logprobs_l = Policy_Model.get_seq_logprobs(prompts, rejected_responses)
    
    # ---------------- 2. 计算 DPO Loss ----------------
    # 计算 policy 与 ref 之间的对数概率差值（隐式奖励）
    implicit_reward_w = policy_logprobs_w - ref_logprobs_w
    implicit_reward_l = policy_logprobs_l - ref_logprobs_l
    
    # 计算 DPO 的对数 Sigmoid 损失
    logits = implicit_reward_w - implicit_reward_l
    loss = - torch.nn.functional.logsigmoid(beta * logits).mean()
    
    # ---------------- 3. 反向传播与参数更新 ----------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

DPO 脱胎于 RLHF with PPO，它通过数学推导证明了它与 PPO 的等价性。同时，DPO 省去了奖励模型，策略模型，甚至省去了环境交互（rollout），它让一切都变成了简单的监督学习（因此，从定义上，DPO并不属于 RL 算法，只是它来自于 RLHF 且与 RLHF 目标相同）。在使用 DPO 进行对齐时，显存里只需要放下 2 个模型（$\pi_\theta$ 和 $\pi_{\text{ref}}$），同时不需要与环境交互，从而极大缩减了时间开销与计算开销。

### GRPO

DPO 简洁而高效的算法来自优雅的理论数学推理；然而，理论与实际之间总有着距离。在实际应用上，DPO 同样有着许多的局限性：

- 首先，由于 DPO 是完全的离线算法，没有采样（rollout）过程，使其非常依赖离线偏好数据的质量，如果数据存在噪声或者分布偏移，DPO 很容易发生过拟合；
- 更大的问题来自 DPO 监督学习的本质。监督学习的本质是模仿，而正如前文在分析 SFT 的局限性时所讲，对于各种复杂的推理问题、数学问题、代码问题，单纯让模型模仿人类的解题过程很难让模型真正理解其背后的逻辑。此时，DPO 这种严重依赖固定偏好对的算法就显得捉襟见肘。

这些问题使得 DPO 在许多情况下依然难以替代庞大的 PPO 算法。那么，能否找到一种算法，能够在不舍弃 PPO 强化学习的优势的同时，又能降低计算开销？Deepseek 团队在 2024 年提出的 [GRPO（Group Relative Policy Optimization，群体相对策略优化）](https://arxiv.org/abs/2402.03300)算法，给出了一个优秀的答案。该算法不仅大幅降低了训练的显存和计算要求，更是后来造就 DeepSeek-R1 强大逻辑推理能力的核心基石。

具体来说，该算法把开销优化的核心放在了价值模型上。重新审视 PPO 算法中价值模型（Critic）的作用：Critic 作为对价值函数 $V_{\pi}(s)$ 的估计，通过 GAE 算法被用于计算优势函数 $A(s,a)$。优势函数的意义是表明当前动作比“平均“动作好多少，其中 Critic 提供的就是那个”平均“动作的分数。

> **直觉**：“平均“动作的分数必须依赖 Critic 进行预测吗？我们能不能回归”平均“二字的本意，用统计学的方法直接对多个动作的奖励进行平均，从而直接得到”平均“动作的分数？

这就是 GRPO 的核心切入点：**基于群体的相对优势（Group Relative Advantage）**：

- 对于同一个输入 Prompt $x$，我们不要只生成一个回答，而是让当前的策略模型并行生成一个群体（Group），包含 $G$ 个不同的回答（比如 $y_1, y_2, \dots, y_G$）。
- 将这 $G$ 个回答丢给裁判（奖励模型或规则校验器），得到 $G$ 个具体的分数 $r_1, r_2, \dots, r_G$。
- 直接使用这群回答的平均分作为”平均”动作的分数，从而计算出每个动作的相对优势。

即，抛弃 Critic 模型，直接在同一 Prompt 生成的群组内部进行标准化（Z-score Normalization）来计算优势函数：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
$$

其中，$\text{mean}(r) = \frac{1}{G}\sum_{i=1}^G r_i$，$\text{std}(r) = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \text{mean}(r))^2}$。

得到优势函数 $\hat{A}_i$ 后，GRPO 的优化目标在形式上与 PPO 非常相似，但它将 PPO 针对序列的计算拉平到了对每一个生成组的期望上。具体来说，GRPO 的目标函数定义为：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim P(x), \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left( \min \left( \rho_{i,t}(\theta) \hat{A}_i, \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}^{i, t} \right) \right]
$$

其中：

1. **$x \sim P(x)$**：从训练集中采样 Prompt。
2. **$\{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}$**：使用旧策略生成 $G$ 个不同的回答。
3. **$\rho_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}$**：重要性采样比率，与 PPO 完全一致。
4. **$\min( \dots , \text{clip}(\dots) )$**：PPO-Clip 的截断机制，限制策略更新步长，防止训练崩溃。
5. **$\beta \mathbb{D}_{\text{KL}}$**：KL 散度惩罚项。与 PPO 把 KL 当作单步奖励（Reward）进行处理不同，由于 GRPO 没有 Critic，而 $\hat{A}_i$ 是归一化的优势，因此显然不能直接将 KL 散度作为奖励塞进优势里。那怎么办呢？只能参考 PPO-Penalty 相同的拉格朗日松弛思想，在 Loss 层面减去 KL 散度。
   但是，直接将原本的 KL 散度惩罚加进 Loss，又会出现问题。我们考虑原本的 KL 散度的导数：
   $$
   \frac{\partial D_{KL}}{\partial \pi_\theta} = \frac{\partial \log \frac{\pi_\theta}{\pi_{\text{ref}}}}{\partial \pi_\theta} = \frac{1}{\pi_\theta}
   $$
   由于 $\pi_\theta > 0$，易知该导数恒正。这意味着，无论当前的 $\pi_\theta$ 是大于 $\pi_{\text{ref}}$ 还是小于 $\pi_{\text{ref}}$，为了最小化这个 Loss，优化器的做法都将是无脑地减小 $\pi_\theta$。这在 $\pi_\theta$ 大于 $\pi_{\text{ref}}$ 时当然很合理，但在 $\pi_\theta$ 小于 $\pi_{\text{ref}}$ 时与我们的意图完全相反。怎么办？DeepSeek 将加入 Loss 中的 KL 惩罚换成了如下公式：
   $$
   \mathbb{D}_{\text{KL}} = \frac{\pi_{\text{ref}}(y_i|x)}{\pi_\theta(y_i|x)} - \log \frac{\pi_{\text{ref}}(y_i|x)}{\pi_\theta(y_i|x)} - 1
   $$
   首先，为什么可以换？我们令 $x$ 为原本的 KL 散度，则该式可以表示为
   $$
   \mathbb{D}_{\text{KL}} =  e^{-x} + x - 1
   $$
   在 GRPO 外层求期望后：
   $$
   \mathbb{E}_{\pi_\theta} [e^{-x} + x - 1] = \mathbb{E}_{\pi_\theta} \left[ \frac{\pi_{\text{ref}}}{\pi_\theta} \right] + \mathbb{E}_{\pi_\theta} [x] - 1 = \underbrace{\sum \left( \pi_\theta \cdot \frac{\pi_{\text{ref}}}{\pi_\theta} \right)}_{= \sum \pi_{\text{ref}} = 1} + \mathbb{E}_{\pi_\theta} [x] - 1 = \mathbb{E}_{\pi_\theta} [x]
   $$
   即，该替换没有改变目标函数。那换完有什么效果？再对其进行求导，得到：
   $$
   \frac{\partial D_{KL}}{\partial \pi_\theta} = \frac{\frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1}{\partial \pi_\theta} = \frac{\pi_\theta - \pi_{\text{ref}}}{\pi_\theta^2}
   $$
   此时可以发现，优化器的更新方向与我们的意图完全相同：永远朝着靠近 $\pi_{\text{ref}}$ 的方向。

**GRPO in LLM 伪代码实现：**

```python
# 初始化
Policy_Model = load_model(SFT_weights)  # 待优化的策略网络 \pi_\theta
Ref_Model = load_model(SFT_weights)     # 冻结的参考网络 \pi_{ref}

Ref_Model.eval()

beta = 0.04        # KL 散度惩罚系数
epsilon = 0.2      # PPO Clip 阈值
G = 8              # 每个 prompt 生成的回答组数（Group size）

for batch_prompts in Dataloader:
    # 假设 batch_prompts 长度为 B
    
    # ---------------- 1. 经验采集阶段 (Rollout) ----------------
    # 将每个 prompt 复制 G 份，形状变为 (B * G)
    repeated_prompts = repeat_prompts(batch_prompts, G)
    
    with torch.no_grad():
        # Policy_Model 生成回答 (使用多核或张量并行加速)
        responses = Policy_Model.generate(repeated_prompts)
        
        # 记录旧策略的生成概率 (Old Logprobs)
        old_logprobs = Policy_Model.get_seq_logprobs(repeated_prompts, responses)
        
        # 获取外部打分 (如果是数学题，这里直接跑验证脚本)
        # rewards 的形状为 (B, G)
        rewards = Evaluator.score(repeated_prompts, responses).view(-1, G)
        
    # ---------------- 2. 计算群体相对优势 (Group Relative Advantage) ----------------
    # 在 G 这个维度上计算均值和标准差
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
    
    # 计算优势函数，并展平回 (B * G)
    advantages = ((rewards - mean_rewards) / std_rewards).view(-1)
    
    # ---------------- 3. GRPO 模型更新阶段 (Optimization) ----------------
    for _ in range(K_epochs):
        # 重新计算当前策略模型的概率
        new_logprobs = Policy_Model.get_seq_logprobs(repeated_prompts, responses)
        
        # 获取冻结参考模型的概率
        with torch.no_grad():
            ref_logprobs = Ref_Model.get_seq_logprobs(repeated_prompts, responses)
        
        # PPO-Clip 计算
        ratio = torch.exp(new_logprobs - old_logprobs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        actor_loss = - torch.min(surr1, surr2)
        
        # 计算 KL 散度惩罚 (DeepSeek 采用的无偏估计)
        # D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
        prob_ratio = torch.exp(ref_logprobs - new_logprobs)
        kl_penalty = prob_ratio - torch.log(prob_ratio) - 1.0
        
        # 最终的损失函数
        loss = (actor_loss + beta * kl_penalty).mean()
        
        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

GRPO 通过引入组相对优势，在保留了 PPO 整体框架的同时，移去了策略模型，从而既保留了 PPO 的各种优势，又成功节省了大量显存；并且由于不需要训练 Critic，训练收敛的速度也明显更快，训练更高效。

### RLVR

最早 InstructGPT 将 PPO 应用到 LLM 后训练中，其出发点是 RLHF，即训练模型的输出与人类的输出对齐。不过，我们在讨论监督学习的局限性时提到的除了难以对齐这一点外，还有另一个难点：解决复杂的推理问题、数学问题、代码问题等问题。这些问题具有以下的共同点：

- 答案往往具是客观确定的，不需要依赖人类主观的“偏好”来评判；
- 结果容易验证，往往规则化的标准就能判断结果的正确性；
- 得到结果前的推导过程往往极为复杂，且难以穷举。

那么各类 RL 算法可以用于提高模型解决这类问题的能力吗？答案是肯定的：我们称这种新的强化学习范式为**RLVR（Reinforcement Learning from Verifiable Rewards，基于可验证奖励的强化学习）**，或基于规则验证的强化学习。

RLVR 与 RLHF 最明显的区别在 Reward 上：RLHF 的奖励来自奖励模型的输出，而 RLVR 的奖励是规则化的。对于代码问题，奖励直接来自代码测试的结果；对于数学问题，奖励则直接来自标准答案与模型输出答案的比对。而这也是 RLVR 的最大优势：绝对客观的奖励输出彻底消除了 RLHF 中的“奖励黑客“问题；同时，由于不依赖通过人类数据训练的奖励模型，让 LLM 超越人类的能力在理论上成为可能。

### CoT

模型想要解决复杂的数学问题、代码问题，显然不可能像一般的对话那样直接输出答案，而是要依靠推理。这就要求模型需要具备**CoT（Chain of Thought，思维链）**。

CoT 的概念最早并不是在 RL 领域提出的，它来自 Google 的论文[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903v6)。在这篇论文中，谷歌的研究人员发现：如果我们在给大模型的示例（Few-shot）中，不只是给出“问题”和“答案”，而是把中间的“推理步骤”也写出来，大模型在解决数学题和逻辑题时的准确率会产生明显的提升。随后在同年的 5 月，东京大学和谷歌的研究者（Takeshi Kojima 等）进一步提出了 Zero-shot CoT，也就是那句"Let's think step by step"（让我们一步一步地思考）。由此可见，最早的触发模型 CoT 能力的方法来自非常简单的 Prompt Engineering。

但单纯依靠提示词触发的 CoT 能力有限。后来，人们通过在 SFT 数据中加入大量由专家标准的包含详细推理步骤的数据，从而提升模型的 CoT 能力。然而，这种做法既昂贵，效果同样有限。

最早将 RL 引入训练模型 CoT 能力的做法来自 OpenAI o1 (2024) 。然而，由于 OpenAI 转入闭源，我们无从得知其具体做法。

终于等到2025年初，Deepseek 带着 Deepseek R1 与论文[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948v2)横空出世。该论文告诉我们：依靠简单的 RL 算法（GRPO）和纯粹的规则验证（RLVR），完全不依赖任何人类提供的高质量思路数据，就能让极其复杂的长思维链（Long CoT）在 LLM 中自然“涌现”。

具体来说，研究员在训练 DeepSeek-R1-Zero 时，跳过了所有 SFT 阶段，直接在纯基座模型（Base Model）上应用强化学习（GRPO）：首先在 system prompt 中要求模型必须先在 \<think> 和 \</think> 标签内写下思考过程，再输出最终答案；然后，加以简洁的奖励设置：

- 准确性奖励：对于数学题和代码题，答案对就给 1 分，错就给 -1 分。
- 格式奖励：只要模型的输出中严格包含了 \<think> 标签和 \</think> 标签，并且把思考过程放在了里面，把结果放在了外面，就能获得加分。

在简单的学习环境下，基座模型在训练过程中自然获得了长 CoT 能力（“**顿悟时刻（Aha Moment）**”）：

- 模型自发地拉长 \<think> 标签内的文本。
- 模型自己学会了反思与自我纠错（Self-Correction）（例如输出：“Wait, this approach seems to lead to a dead end, let me try another way...”）。
- 模型甚至学会了针对同一个问题分配多种不同的解法进行交叉验证。

这样的实验结果实在令人振奋：它充分展现了 RLVR 的强大与 LLM 在可验证任务上的巨大潜力。从此，各类针对 RLVR 的 RL 算法改进开始如雨后春笋般冒出，LLM 的 math/coding/agent 能力也开始飞跃。

### DAPO

DeepSeek-R1 证明了纯 RL 在推理任务（RLVR）上的巨大潜力，但由于其部分底层技术细节（如具体的 PPO/GRPO 调参和数据流控制）并未完全开源，研究者在复现超长思维链（Long-CoT）RL 时经常遭遇训练极不稳定、探索停滞等问题。

2025 年 3 月，字节跳动与清华大学等机构联合提出了 **[DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，解耦截断与动态采样策略优化）](https://arxiv.org/abs/2503.14476)**，对 GRPO 算法进行改进。该论文指出，GRPO 算法在训练中存在以下缺陷：

- 熵崩溃（entropy collapse）：随着训练进行，策略的熵值急剧下降，模型生成的多样样本迅速减少，导致探索能力枯竭。
- 奖励噪声：特别是在处理长思维链（CoT）时，因响应过长而被截断的样本会收到惩罚性奖励，这混淆了模型对正确推理过程的判断。
- 训练不稳定与低效：当一批样本全部回答正确或全部错误时，其优势函数（Advantage）为零，导致梯度信号消失，浪费计算资源并引入噪声。

针对这些缺陷，DAPO 对 GRPO 算法做出了多项修改：

- **移除 KL 惩罚**：KL 惩罚项用于调节在线策略与冻结参考策略之间的差异。在 RLHF 场景中，强化学习的目标是使模型行为与初始模型保持一致，避免过度偏离。然而，在训练长 CoT 推理模型的过程中，模型分布可能与初始模型产生显著偏差，因此这种限制不再必要。

- **Clip-Higher 策略**：将 PPO/GRPO 中统一的 clipping 范围解耦为上下两个阈值 $\epsilon_{low}$ 和 $\epsilon_{high}$，并提高上界 $\epsilon_{high}$（一般改为0.28）。这样可以为低概率 token 的概率提升提供更大的空间，从而增强策略探索能力，缓解训练过程中策略熵崩溃的问题，同时保持较小的下界以避免概率被压到接近 0 导致采样空间塌缩。

- **动态采样（Dynamic Sampling）**：在采样阶段进行过采样，并过滤掉“全部正确”或“全部错误”的样本组（即 accuracy 为 1 或 0 的组）。因为这类样本组的奖励完全相同，优势函数为 0，会导致策略梯度为 0，从而浪费训练计算。通过持续采样直到 batch 中只包含具有非零优势的样本组，可以保持有效梯度信号，提高训练稳定性与样本效率。

- **Token 级损失**：将原 GRPO 的 sample-level loss 改为 token-level policy gradient loss。GRPO 会先对每个样本的 token loss 求平均再在样本间求平均，导致长序列中的 token 对整体梯度贡献被稀释。在长 CoT 场景中，这会削弱模型对高质量长推理的学习能力，并难以惩罚冗长或重复的低质量生成。Token 级损失让每个 token 对梯度贡献更加均衡，使长推理序列能够提供更充分的训练信号。

- **软过长惩罚**：针对超过最大生成长度而被截断的样本，引入长度感知的奖励 shaping。传统做法通常直接给予强惩罚，这可能误伤本身正确但较长的推理过程，带来奖励噪声。DAPO 在接近最大长度的区间内施加随长度逐渐增加的惩罚，并在真正超出上限时才给予最大惩罚，从而减少噪声并稳定训练。

在加入以上改进后，DAPO 算法的公式变为：

$$
\mathcal{J}_{\mathrm{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim \mathcal{D},\{o_i\}_{i=1}^{G}\sim \pi_{\theta_{\mathrm{old}}}(\cdot|q)}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{clip}\left(r_{i,t}(\theta),1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}\right)\hat{A}_{i,t}\right)\right]
$$

约束条件（Dynamic Sampling）：

$$
0 < \left|\{o_i \mid \mathrm{is\_equivalent}(a,o_i)\}\right| < G
$$

重要性采样比率与优势函数与GRPO相同：

$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,\lt t})}
$$

$$
\hat{A}_{i,t}=\frac{R_i-\mathrm{mean}(\{R_i\}_{i=1}^{G})}{\mathrm{std}(\{R_i\}_{i=1}^{G})}
$$

但奖励函数变为：

$$
R_{\text{final}}(y) = R_{\text{correct}}(y) + R_{\text{length}}(y)
$$

其中 $R_{\text{length}}(y)$ 是软过长惩罚：

$$
R_{\mathrm{length}}(y)=\begin{cases}0, & |y|\le L_{\mathrm{max}}-L_{\mathrm{cache}} \\\frac{(L_{\mathrm{max}}-L_{\mathrm{cache}})-|y|}{L_{\mathrm{cache}}},& L_{\mathrm{max}}-L_{\mathrm{cache}}<|y|\le L_{\mathrm{max}} \\-1, & |y|>L_{\mathrm{max}}\end{cases}
$$

### CISPO

前文提到的 GRPO 训练存在的问题中，影响最严重的是熵崩溃，这会导致训练直接崩溃。DAPO 尝试通过加大截断上限（Clip-Higher）来鼓励探索，从而缓解熵崩溃的问题，但这样的做法显得有些流于表面。MiniMax 团队在2025年6月发布模型 [MiniMax-M1](https://arxiv.org/abs/2506.13585) 时，针对这个问题进行了更进一步的分析和改进。

在论文中，研究员指出：GRPO（以及 PPO）中的 clipping 机制会在 RL 训练早期直接“剪掉”一部分关键 token 的梯度，从而破坏探索能力并导致熵快速下降。具体来说，又前文可知，在 PPO / GRPO 的目标函数中，每个 token 的更新都会乘以一个重要性采样权重 $r_{i,t}(θ)$，这个权重用于在 off-policy 更新时修正分布偏差，然后为了防止更新过大，PPO 引入了 clipping 操作：

$$
min(r_{i,t} Â_{i,t}, clip(r_{i,t}, 1-ε, 1+ε) Â_{i,t})
$$

clip 可以限制策略更新幅度，从而稳定训练。然而在实际的大模型推理训练中，研究人员发现它会带来一个意想不到的问题。

在推理模型的生成过程中，一些具有 **反思或分叉意义的 token**（例如 However、Wait、Recheck、Aha 等）通常在 base model 中概率较低。这意味着当模型在 RL 过程中第一次尝试生成这些 token 时，新的策略概率 $π_θ$ 往往会比旧策略 $π_{θ_old}$ 高得多，从而导致对应的 $r_{i,t}$ 取值很大。

一旦 $r_{i,t}$ 超过 clipping 上限，这些 token 在 PPO/GRPO 的目标函数中就会被截断，直接失去梯度贡献。由于 RL 训练通常会对同一批生成数据进行多轮 off-policy 更新，这些 token 在第一次更新后就被“剪掉”，后续更新也无法再从它们获得梯度信号。

问题在于，这些低概率 token 往往恰恰是 Long CoT 中的关键结构节点。它们常常对应着推理路径中的分支、反思或自我修正。如果这些 token 的梯度被持续裁剪，模型就会逐渐倾向于生成更加确定、但更短、更单一的推理路径，从而导致探索能力下降以至熵崩溃。

为了解决这一问题，MiniMax 团队提出了 **CISPO（Clipped Importance Sampling Policy Optimization）**。其核心思想是：**不再裁剪 token 的梯度，而是只裁剪重要性采样权重本身**。再结合 DAPO 提到的 token 级损失，得到 CISPO 的目标函数：

$$
\mathcal{J}_{CISPO}(θ) = E_{(q,a) \sim D, {o_i}_{i=1}^G \sim π_{θ_{old}}(·|q)}  \left[ \frac{1}{\sum_{i=1}^G |o_i|}  ·  ∑_{i=1}^G ∑_{t=1}^{|o_i|} sg( r̂_{i,t}(θ) ) Â_{i,t} log π_θ(o_{i,t} | q, o_{i,\lt t}) \right]
$$

其中，sg(·) 表示 **stop-gradient 操作**（即把其中变量看成常量计算梯度），而

$$
\hat{r}_{i,t}(θ) = clip( r_{i,t}(θ), 1 − ε_{low}^{IS}, 1 + ε_{high}^{IS} )
$$

让我们分析此目标函数，与 PPO 进行比对：

- 在CISPO 中，对 $r_{i,t}$ 做 **clip** 得到 $r̂_{i,t}$，并通过 **stop-gradient** 阻断其梯度传播，最终每个 token 的梯度形式为：
  $$
  ∇_θ J_{\text{CISPO}} ≈ r̂_{i,t} Â_{i,t} ∇_θ \log π_θ(o_{i,t}|…)
  $$
  比率 $\hat{r}_{i,t}$ 被限制在上下限内，但永远不为 0，每个 Token 始终在提供梯度更新的推力。
- 而在 PPO 中，单 Token 梯度公式为：
  $$
  \nabla_\theta J_{\text{PPO}} \approx \mathbb{I}_{i,t} \cdot r_{i,t} \hat{A}_{i,t} \nabla_\theta \log \pi_\theta(o_{i,t}|\dots)
  $$
  其中，$\mathbb{I}_{i,t}$ 是一个指示函数，用于判断当前 Token 是否触发了 PPO 的截断规则：
  $$
  \mathbb{I}_{i,t} =\begin{cases}0, & \text{如果 } (r_{i,t} > 1+\epsilon \text{ 且 } \hat{A}_{i,t} > 0) \text{ 或 } (r_{i,t} < 1-\epsilon \text{ 且 } \hat{A}_{i,t} < 0) \\1, & \text{其他情况（未触发截断）}\end{cases}
  $$
  一旦重要性比率超限，指示函数 $\mathbb{I}_{i,t}$ 变为 0，当前 Token 的梯度被物理掐断。

因此，CISPO 既保留了 PPO-Clip 降低梯度方差的效果，又避免了 clipping 导致的梯度贡献丢失、模型探索能力下降的问题，从而带来了更好的训练效果。同时，由于每个 Token 都始终在为梯度更新做贡献，CISPO 也有着显著更高的训练效率。

> 另外需要注意：CISPO 修改了重要性采样的公式（加入了截断），这种做法会引入一定程度的梯度偏差。但论文的实验表明，这种偏差的影响远小于 token clip 带来的训练损害。

### GSPO

在 GRPO 容易熵崩溃这个问题上，Minimax 的研究员认为其原因来自 clipping 机制导致的梯度贡献丢失，于是移去 clipping 机制，转而直接裁剪重要性采样权重，从而得到 CISPO；而 Qwen 团队认为，这依然不是最根本的原因。在2025年7月发表的论文[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071v2) 中，Qwen 的研究员将问题的根本原因定位为：**优化目标的单位和奖励的单位不一致**。

具体来说，在 GRPO 中，奖励是针对完整序列给出的，但优化却是在 token 层面进行的。在 RLHF 或 reasoning 任务中，一个生成结果 $y$ 的 reward $r(x,y)$ 是在整个序列生成完之后才计算的。例如答案是否正确、推理是否完成、格式是否满足要求等，这些评价都依赖完整响应。因此 reward 的自然单位是 sequence-level。

然而 GRPO 在优化时，却将这个 sequence-level 的优势 $\hat{A}_i$ 平均分配到每一个 token 上，并在每个 token 上分别应用重要性采样权重 $r_{i,t}$。问题在于：这个 token-level 的重要性权重是基于单个采样 token 计算的，而不是基于整个 token 分布的期望。当重要性采样只基于单样本时，它无法有效修正行为策略与目标策略之间的分布偏差，反而会引入高方差噪声。在长序列生成任务中，这种噪声会在 token 维度不断累积。

进一步地，GRPO 还会对这些 token-level 权重应用 clipping。由于每个 token 的权重都不同，这些不均匀的权重在序列中不断叠加，可能导致梯度方向变得不可预测，从而引发训练不稳定，导致出现训练崩溃。

那如何解决这个问题？

> 直觉：既然奖励是在 sequence level 上定义的，那么 重要性采样也应该在 sequence level 上进行，而不是在 token level 上逐个应用。

基于此想法，论文提出 **GSPO（Group Sequence Policy Optimization）**，其核心思想正是将 GRPO 的 token-level 优化目标改为 sequence-level 优化目标。具体来说，是直接对整个响应 $y$ 计算序列级的重要性采样权重：

$$
s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{\frac{1}{|y_i|}}
$$

而序列概率可以分解为逐 token 的条件概率：

$$
\pi_\theta(y_i|x)=\prod_{t=1}^{|y_i|}\pi_\theta(y_{i,t}|x,y_{i,\lt t})
$$

因此可以将上式展开为对数形式：

$$
s_i(\theta)=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,\lt t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,\lt t})}\right)
$$

这里的权重衡量的是：当前策略生成该完整序列的概率，相对于旧策略发生了多大的变化。同时论文对该比值进行了长度归一化（ $1/|y_i|$ ），以避免长序列导致的数值爆炸，并降低方差。

随后，GSPO 将 clipping 也应用在整个序列的权重上，而不是 token 上：

$$
\min\left(s_i(\theta)\hat A_i,\; \text{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\hat A_i\right)
$$

从而得到 GSPO 的目标函数：

$$
\mathcal{J}_{\mathrm{GSPO}}(\theta)=\mathbb{E}_{x\sim \mathcal{D}, \{y_i\}_{i=1}^{G}\sim \pi_{\theta_{\mathrm{old}}}(\cdot |x)}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\widehat{A}_i,\mathrm{clip}\left(s_i(\theta),1-\epsilon,1+\epsilon\right)\widehat{A}_i\right)\right]
$$

这样一来，重要性采样在 sequence level 进行，clipping 也在 sequence level 进行，一个序列内部的所有 token 共享同一个权重，从而使得优化粒度与 reward 粒度保持一致，避免了 GRPO 中 token 权重不均带来的梯度噪声累积，进而显著降低方差并提升训练稳定性。

## 结语：在数学严谨与工程妥协中演进的强化学习

回顾大语言模型中强化学习的发展脉络，我们可以清晰地看到一条在“数学严谨”与“工程妥协”之间不断寻找最优解的实用主义路径。

从底层逻辑来看，所有的算法演进都在试图回答一个核心问题：**如何在一个拥有数百亿参数、输出极其绵长的动作空间中，将稀疏的奖励信号稳定且高效地分配给每一个 Token。**

REINFORCE 给出了最原始的方向，PPO 则通过引入 Critic 模型和 PPO-Clip 机制，确立了在线强化学习在稳定性上的黄金标准，促成了 RLHF 的繁荣；

DPO 洞察到了 KL 惩罚下的闭式解，用优雅的数学等价替换了复杂的在线采样与奖励拟合，极大地降低了偏好对齐的门槛；

当战线推进到深度推理（System 2）与超长思维链（Long-CoT）时，由于需要极高的探索空间，**RLVR（基于可验证奖励的强化学习）** 登上了历史舞台。GRPO 果断砍掉了臃肿的 Critic 模型，通过群体相对优势实现了算力的彻底解放，促成了模型逻辑能力的自然涌现；

面对长文本带来的梯度方差与截断失效，DAPO 重新设计了截断策略，CISPO 利用停止梯度（Stop-Gradient）挽救了死区 Token，而 GSPO 则完成了从微观 Token 级采样到宏观序列级打包的降维打击。

每一次算法的更迭，本质上都是对显存、算力、梯度方差以及数据利用率的重新平衡。

预训练（Pre-training）构建了大模型的知识广度与概率底座，监督微调（SFT）赋予了其遵循指令与人类交互的规范。而如今，强化学习（RL）已不再仅仅是出厂前的最后一道“抛光工序”，它正在褪去模仿人类的表象，成为驱动大语言模型在客观规律中自我博弈、突破静态数据上限的核心引擎。

时至今日，仍然有各种改进的强化学习算法源源不断地涌现，模型的能力也水涨船高，甚至在程序开发领域已经引发了大规模的恐慌。但与此同时，我们也需要意识到，现阶段的强化学习并非十全十美。在面对难以验证的任务时，各种算法仍旧依赖奖励模型，而奖励模型天然的缺陷致使强化学习依然步履维艰，依然在等待着新的突破。山的那头是什么？何处是RL的边界？仍未可知。

## References

- [1] Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). *arXiv preprint arXiv:1707.06347*.
- [2] Wei, J., et al. (2022).[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). *arXiv preprint arXiv:2201.11903*.
- [3] Ouyang, L., et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). *arXiv preprint arXiv:2203.02155*.
- [4] Rafailov, R., et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). *arXiv preprint arXiv:2305.18290*.
- [5] Shao, Z., et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300). *arXiv preprint arXiv:2402.03300*.
- [6] Guo, D., et al. (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). *arXiv preprint arXiv:2501.12948*.
- [7] Zhang, Y., et al. (2025).[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476). *arXiv preprint arXiv:2503.14476*.
- [8] Wang, X., et al. (2025).[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585). *arXiv preprint arXiv:2506.13585*.
- [9] Li, H., et al. (2025). [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071). *arXiv preprint arXiv:2507.18071*.

## Citation

如果你在研究或工作中引用了本文，请以以下格式引用：

**BibTeX:**

```bibtex
@misc{long2026rl_llm,
  author       = {Long, Yijun},
  title        = {大模型中的强化学习：从 REINFORCE 到 xxPO},
  year         = {2026},
  howpublished = {\url{https://procrastinatorrrr.github.io/posts/tech/202604-rl-llm/}},
  note         = {Accessed: 2026-04-21}
}
```

**APA Style:**

```txt
Long, Y. (2026). 大模型中的强化学习：从 REINFORCE 到 xxPO. https://procrastinatorrrr.github.io/posts/tech/202604-rl-llm/
```
