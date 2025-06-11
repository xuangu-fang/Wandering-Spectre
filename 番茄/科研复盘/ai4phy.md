

## iclr-2024

[Physics-informed neural networks for sampling](https://openreview.net/forum?id=KwHPBIGkET) 
[Learning Stochastic Dynamics from Data](https://openreview.net/forum?id=MdXtFDhy0H)


## iclr-2025
#### MeshMask: Physics-Based Simulations with Masked Graph Neural Networks

[https://openreview.net/forum?id=bFHR8hNk4I](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DbFHR8hNk4I)

提出了一种新的掩码预训练技术，应用于图神经网络（GNN）以解决计算流体动力学（CFD）问题。通过随机掩码输入网格中最多40%的节点，该方法迫使模型学习复杂流体动力学的鲁棒表示，并结合非对称编码器-解码器架构和门控多层感知器（MLP）进一步提升性能。论文在七个CFD数据集上实现了先进的结果，包括一个具有超过25万个节点的3D颅内动脉瘤模拟数据集，显著提高了长期预测准确性和训练效率。

**“物理场也有自我修养：自监督预训练”**

#### PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations

[https://openreview.net/forum?id=y5B0ca4mjt](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3Dy5B0ca4mjt)

该论文提出了一种新的物理信息高斯（PIG）方法，用于近似求解偏微分方程（PDE）。与传统基于多层感知机（MLP）的物理信息神经网络（PINNs）相比，PIG通过结合高斯特征嵌入和轻量级神经网络，能够动态调整高斯的位置和形状，从而更有效地捕捉高频率和非线性成分。实验结果表明，PIG在多个PDE基准测试中表现出色，展示了其作为高效PDE求解器的潜力。

**“PINN的精神继承”**

  

#### Physics-Informed Diffusion Models

[https://openreview.net/forum?id=tpYeermigp](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DtpYeermigp)

论文提出了一种将扩散模型（Diffusion Models）与物理约束（Partial Differential Equations，PDEs）相结合的框架，称为“物理信息扩散模型”（Physics-Informed Diffusion Models, PIDM）。该框架通过在训练过程中引入基于物理定律的损失项，确保生成的样本满足相应的物理约束。实验结果表明，PIDM 在流体流动和结构拓扑优化等任务中显著提升了性能，降低了 PDE 残差，并表现出对抗过拟合的有效性。

**“不合理就惩罚下，扩散模型加点物理损失”**


#### PIED: Physics-Informed Experimental Design for Inverse Problems

[https://openreview.net/forum?id=w7P92BEsb2](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3Dw7P92BEsb2)

论文提出了一个新的实验设计（ED）框架，称为**物理信息实验设计（PIED）**，用于解决由偏微分方程（PDE）控制的系统中的逆问题。PIED通过使用物理信息神经网络（PINN）作为前向模拟器和逆求解器，优化设计参数（如传感器放置），以在有限的观测预算下最大化信息量。PIED克服了现有方法在计算效率的瓶颈，并通过并行计算和元学习优化PINN参数初始化。

重建做得差不多就该做测点优化了，除了物理场，还有激光雷达呢：

> [1] Li Y, Kong L, Hu H, Xu X, Huang X. Is Your LiDAR Placement Optimized for 3D Scene Understanding?[C]//The Thirty-eighth Annual Conference on Neural Information Processing Systems. , 2024.

**“有重建的地方就有传感器，有传感器的就得布传感器”**
  

  #### [Physics-Informed Deep Inverse Operator Networks for Solving PDE Inverse Problems](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3D0FxnSZJPmh)**

标签：Neural Operator, Unsupervised

**“PI-DIONs 在算子学习框架中引入稳定性估计，实现无监督 PDE 逆问题求解，确保稳定高效的实时推理。”**


**[PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3Dy5B0ca4mjt)**

标签：High-Frequency, PINN

**“PIGs 通过可学习的高斯特征嵌入动态调整参数网格，在提高 PDE 求解精度的同时，实现更高效的计算。”**

  
   **[Physics-Informed Neural Predictor](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DvAuodZOQEZ)**

标签：Fluid, Multi-Physics, PINN

**“本文将物理方程嵌入神经预测器，实现了流体动力学的高精度长期预测，并具备出色的时空泛化能力。”**


**[Physics-aligned field reconstruction with diffusion bridge](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DD042vFwJAM)**

标签：Boundary

**“PalSB 采用物理对齐的扩散桥机制，从稀疏观测数据重建物理场，提升精度并确保物理一致性。”**

“PalSB employs a physics-aligned diffusion bridge to reconstruct physical fields from sparse measurements, achieving higher accuracy and compliance with physical constraints.”

**[PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DfU8H4lzkIm)**

标签：Boundary, GNN, Temporal

**“PhyMPGN 结合物理感知消息传递与拉普拉斯算子，在不规则网格上实现高精度时空 PDE 建模。”**

“PhyMPGN embeds physics-aware message passing and Laplacian operators into a graph network, enabling accurate spatiotemporal PDE modeling on irregular meshes.”

  
  **[Gradient-Free Generation for Hard-Constrained Systems](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DteE4pl9ftK)**

标签：Gradient-Free, Hard Constraints, Zero-Shot

**“本文提出了一种梯度无关的零样本生成框架，确保PDE系统中的硬约束严格满足，同时保持分布的准确性。”**

“This paper introduces a gradient-free, zero-shot generative sampling framework that enforces hard constraints in PDE systems while preserving distribution accuracy.”

**[Generating Physical Dynamics under Priors](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DeNjXcP6C0H)**

标签：Diffusion, PINN, Physical Priors

**“本文提出了一种融合物理先验的扩散生成模型，以生成符合物理规律的动态，提高了仿真精度和真实感。”**

“This paper presents a diffusion-based generative model incorporating physical priors to generate physically feasible dynamics, enhancing realism and accuracy.”

  **[ANaGRAM: A Natural Gradient Relative to Adapted Model for efficient PINNs learning](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3Do1IiiNIoaA)**

标签：Functional Analysis, Green’s Function, Optimization, PINN

**“ANaGRAM 通过结合泛函分析与格林函数理论，引入低复杂度的自然梯度优化方法，有效提升 PINN 的训练效率与稳定性。”**

"ANaGRAM enhances PINN training efficiency by introducing a reduced-complexity natural gradient optimization method grounded in functional analysis and Green’s functions."

  

  