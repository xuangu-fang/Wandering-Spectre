
# Part I — 生成式建模（Diffusion ↔ Flow Matching，含逆问题）

## 1) 必须掌握的概念

- 噪声调度与反向去噪：理解前向加噪、反向去噪如何逼近真实数据分布；决定采样步数与稳定性（DDPM 基本盘）。([arXiv](https://arxiv.org/abs/2006.11239?utm_source=chatgpt.com "[2006.11239] Denoising Diffusion Probabilistic Models"), [NeurIPS 会议记录](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf?utm_source=chatgpt.com "Denoising Diffusion Probabilistic Models"))
    
- 隐式采样与加速：在不改训练目标的前提下，把多步马尔可夫采样变成确定性少步推断（DDIM 的价值）。([arXiv](https://arxiv.org/abs/2010.02502?utm_source=chatgpt.com "[2010.02502] Denoising Diffusion Implicit Models"))
    
- 连续时间统一视角：用 SDE/ODE 把扩散与流联系起来，统一训练与采样框架，为“流匹配”打地基。([arXiv](https://arxiv.org/abs/2011.13456?utm_source=chatgpt.com "Score-Based Generative Modeling through Stochastic Differential Equations"))
    
- 逆问题与后验采样：把观测算子与噪声模型并进扩散后验采样，实现成像/物理重建中的测量一致性。([arXiv](https://arxiv.org/abs/2209.14687?utm_source=chatgpt.com "Diffusion Posterior Sampling for General Noisy Inverse ..."), [ICLR](https://iclr.cc/virtual/2023/poster/11877?utm_source=chatgpt.com "Diffusion Posterior Sampling for General Noisy Inverse ..."))
    
- Flow Matching/Rectified Flow：直接回归速度场，训练连续归一化流，采样少步、收敛稳，适合物理场生成与约束。([arXiv](https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "Flow Matching for Generative Modeling"))
    
- [optional] 薛定谔桥（SB）与统一框架：SB、流匹配、扩散在“概率路径/插值”下互通，便于把物理约束并入生成轨迹。([arXiv](https://arxiv.org/abs/2303.16852?utm_source=chatgpt.com "[2303.16852] Diffusion Schrödinger Bridge Matching"))
    

## 2) 经典必读（排序分先后）

1. **Denoising Diffusion Probabilistic Models**，Jonathan Ho et al., NeurIPS 2020  
    为什么：最小可用实现，MSE 目标易训，弄清“加噪/去噪”的所有符号与记法；后续一切方法的坐标原点。([arXiv](https://arxiv.org/abs/2006.11239?utm_source=chatgpt.com "[2006.11239] Denoising Diffusion Probabilistic Models"), [NeurIPS 会议记录](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf?utm_source=chatgpt.com "Denoising Diffusion Probabilistic Models"))
    
2. **Denoising Diffusion Implicit Models**，Jiaming Song et al., 2020  
    为什么：保持原训练，推断改为确定性常微分离散，采样速度上来；帮学生建立“训练与采样可分离”的直觉。([arXiv](https://arxiv.org/abs/2010.02502?utm_source=chatgpt.com "[2010.02502] Denoising Diffusion Implicit Models"))
    
3. **Score-Based Generative Modeling through Stochastic Differential Equations**，Yang Song et al., ICLR 2021  
    为什么：统一 SDE/ODE 视角与 PC 采样器，打开“把扩散当数值积分器调优”的门。([arXiv](https://arxiv.org/abs/2011.13456?utm_source=chatgpt.com "Score-Based Generative Modeling through Stochastic Differential Equations"))
    
4. **Diffusion Posterior Sampling for General Noisy Inverse Problems**，Hyungjin Chung et al., ICLR 2023
    为什么：把扩散与测量模型拼起来做后验采样，天然适配物理反演。([arXiv](https://arxiv.org/abs/2209.14687?utm_source=chatgpt.com "Diffusion Posterior Sampling for General Noisy Inverse ..."), [OpenReview](https://openreview.net/forum?id=OnD9zGAGT0k&utm_source=chatgpt.com "Diffusion Posterior Sampling for General Noisy Inverse ..."), [ICLR](https://iclr.cc/virtual/2023/poster/11877?utm_source=chatgpt.com "Diffusion Posterior Sampling for General Noisy Inverse ..."))  
    
    
5. **Flow Matching for Generative Modeling**，Yaron Lipman et al., 2022→NeurIPS 2023 教程资源  
    为什么：无仿真训练 CNF，路径可选（含 OT 路径），速度快、稳定，对小步数更友好。([arXiv](https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "Flow Matching for Generative Modeling"), [NeurIPS](https://neurips.cc/virtual/2024/tutorial/99531?utm_source=chatgpt.com "NeurIPS Tutorial Flow Matching for Generative Modeling"))
    
6. **Rectified Flow: Learning to Generate and Transfer Data with Rectified Flow**，Xingchao Liu et al., ICLR 2023  
    为什么：把轨迹“拉直”，极少步采样，易做一/几步生成与蒸馏，工程上很好调。([arXiv](https://arxiv.org/abs/2209.03003?utm_source=chatgpt.com "Learning to Generate and Transfer Data with Rectified Flow"), [OpenReview](https://openreview.net/revisions?id=XVjTT1nw5z&utm_source=chatgpt.com "Revision History for Flow Straight and Fast: Learning to..."))
    

## 3) 推荐开源库与可跑 demo

- **Hugging Face diffusers**：训练/推断/调度器齐全，改采样器最省事；课程用它跑 DDPM/DDIM 即可。文档与课程也齐。([GitHub](https://github.com/huggingface/diffusers?utm_source=chatgpt.com "State-of-the-art diffusion models for image, video, and ..."), [Hugging Face](https://huggingface.co/docs/diffusers/en/index?utm_source=chatgpt.com "Diffusers"))
    
- **score_sde_pytorch**（Yang Song 官方）：SDE 版扩散的标准实现，含 PC 采样与逆问题示例。([GitHub](https://github.com/yang-song/score_sde?utm_source=chatgpt.com "Official code for Score-Based Generative Modeling through ..."))
    
- **facebookresearch/flow_matching**：Flow Matching 官方库，API、教程、notebook 现成，便于上手路径与速度场回归。([GitHub](https://github.com/facebookresearch/flow_matching?utm_source=chatgpt.com "A PyTorch library for implementing flow matching ..."), [Facebook Research](https://facebookresearch.github.io/flow_matching/?utm_source=chatgpt.com "Flow Matching documentation"))
    
- **torchsde / torchdiffeq**：SDE/ODE 可微积分器，做自定义采样器与物理耦合时必备。([GitHub](https://github.com/google-research/torchsde?utm_source=chatgpt.com "google-research/torchsde: Differentiable SDE solvers with ..."))
    
- 逆问题参考实现：**DDRM**（线性问题零样本），[**InverseBench**]([InverseBench: Benchmarking Plug-and-Play Diffusion Models for Inverse Problems in Physical Sciences](https://devzhk.github.io/InverseBench/#:~:text=With%20InverseBench%2C%20we%20benchmark%2014%20inverse%20problem%20algorithms,into%20the%20strengths%20and%20weaknesses%20of%20existing%20algorithms.))（多科学逆问题基准，可比对 DPS/Plug-and-Play 等）。([GitHub](https://github.com/bahjat-kawar/ddrm?utm_source=chatgpt.com "bahjat-kawar/ddrm: [NeurIPS 2022] Denoising Diffusion ..."), [Denoising Diffusion Restoration Models](https://ddrm-ml.github.io/?utm_source=chatgpt.com "Denoising Diffusion Restoration Models"))
    

---

# Part II — 物理 AI 模型（Operator Learning：PINN ↔ FNO 等）

## 1) 必须掌握的概念

- PDE 基础与数值解：初边值问题、稳定性/收敛性、谱方法直觉；决定你的网络输出是否“像个解”。
    
- PINN 与神经算子之别：PINN 用残差在点上“惩罚”方程，神经算子直接学“函数到函数”的映射，跨域泛化强。
    
- 物理约束嵌入：软约束（loss 加 PDE 残差）、硬约束（坐标变换或特征层内置守恒/边界）。
    
- 泛化与外推：跨参数、跨网格、跨几何；评估要分清 in-distribution 与 out-of-distribution。
    

## 2) 经典必读（按学习顺序）

1. **Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear PDEs**，Raissi, Perdikaris, Karniadakis, JCP 2019（两篇 2017–2019 系列）  
    为什么：最清晰的“把物理写进损失”的范式；先用它在小问题上练手，再认识其局限。([arXiv](https://arxiv.org/abs/1711.10561?utm_source=chatgpt.com "Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations"), [OSTI](https://www.osti.gov/pages/biblio/1595805?utm_source=chatgpt.com "Physics-informed neural networks: A deep learning ..."))
    
2. **Fourier Neural Operator for Parametric Partial Differential Equations**，Zongyi Li et al., arXiv 2020 → ICLR 2021  
    为什么：在频域参数化核，跨域/跨分辨率泛化强，是最实用的神经算子之一。([arXiv](https://arxiv.org/abs/2010.08895?utm_source=chatgpt.com "Fourier Neural Operator for Parametric Partial Differential Equations"), [Columbia Leap](https://leap.columbia.edu/wp-content/uploads/2023/01/Li-et-al.2021.pdf?utm_source=chatgpt.com "arXiv:2010.08895v3 [cs.LG] 17 May 2021"))
    
3. **DeepONet: Learning Nonlinear Operators Based on the Universal Approximation Theorem of Operators**，Lu, Jin, Karniadakis, 2019→Nature MI 2021  
    为什么：通用结构（Branch/Trunk），作为 FNO 的互补基线，适于小数据/多几何。([arXiv](https://arxiv.org/abs/1910.03193?utm_source=chatgpt.com "DeepONet: Learning nonlinear operators for identifying differential ..."), [OSTI](https://www.osti.gov/biblio/2281727?utm_source=chatgpt.com "Learning nonlinear operators via DeepONet based on the universal ..."))
    
4. **Choose a Transformer: Fourier or Galerkin（Galerkin Transformer）**，Shuhao Cao, NeurIPS 2021  
    为什么：把注意力机制引入算子学习，展示“谱卷积 + 线性注意力”的高效组合。([NeurIPS 会议记录](https://proceedings.neurips.cc/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html?utm_source=chatgpt.com "Choose a Transformer: Fourier or Galerkin"), [OpenReview](https://openreview.net/pdf?id=ssohLcmn4-r&utm_source=chatgpt.com "Choose a Transformer: Fourier or Galerkin"))
    

## 3) 推荐开源库与可跑 demo

- **neuraloperator**（官方 FNO/TNO）：含 Burgers、Darcy、Navier-Stokes 等脚本与教程。([GitHub](https://github.com/neuraloperator/neuraloperator?utm_source=chatgpt.com "Learning in infinite dimension with neural operators."), [Neural Operator](https://neuraloperator.github.io/?utm_source=chatgpt.com "Neural Operators in PyTorch — neuraloperator 1.0.2 ..."))
    
- **DeepXDE**：PINN 框架，示例涵盖 Allen-Cahn、Navier-Stokes、IDE 等。([GitHub](https://github.com/lululxvi/deepxde?utm_source=chatgpt.com "lululxvi/deepxde: A library for scientific machine learning ..."), [DeepXDE](https://deepxde.readthedocs.io/?utm_source=chatgpt.com "DeepXDE — DeepXDE 1.14.1.dev8+gb944422 documentation"))
    
- **NVIDIA PhysicsNeMo / Modulus**：工程化物理 AI 管线，分布式训练与硬约束工具链。([GitHub](https://github.com/NVIDIA/physicsnemo?utm_source=chatgpt.com "NVIDIA/physicsnemo: Open-source deep-learning ..."), [NVIDIA Developer](https://developer.nvidia.com/physicsnemo?utm_source=chatgpt.com "NVIDIA PhysicsNeMo"))
    
- **Galerkin Transformer 实现**：线性注意力算子学习 demo。([GitHub](https://github.com/scaomath/galerkin-transformer?utm_source=chatgpt.com "[NeurIPS 2021] Galerkin Transformer: a linear attention ..."))
    

---

# 数据与任务基准（物理场生成/预测）

- **PDEBench（NeurIPS 2022 Datasets & Benchmarks）**：提供 Burgers、Advection、Reaction-Diffusion、Darcy Flow、不可压 Navier-Stokes、浅水方程等，含数据生成与下载脚本、FNO/UNet/PINN 基线与训练脚本。([GitHub](https://github.com/pdebench/PDEBench "GitHub - pdebench/PDEBench: PDEBench: An Extensive Benchmark for Scientific Machine Learning"))
    
    - pip 安装与数据下载脚本、HDF5 合并、基线训练入口 `train_models_forward.py / inverse.py` 都在仓库 README。([GitHub](https://github.com/pdebench/PDEBench "GitHub - pdebench/PDEBench: PDEBench: An Extensive Benchmark for Scientific Machine Learning"))
        

---

# 课堂/自学路线图（8 个里程碑）

1. 复现实验：DDPM→DDIM，在 CIFAR-10 或小型 2D 物理场块上跑 50→20 步采样对比。([arXiv](https://arxiv.org/abs/2006.11239?utm_source=chatgpt.com "[2006.11239] Denoising Diffusion Probabilistic Models"))
    
2. 换连续视角：跑 score_sde_pytorch 的 VESDE/CESDE + PC 采样器，对比 ODE/EM 采样。([GitHub](https://github.com/yang-song/score_sde?utm_source=chatgpt.com "Official code for Score-Based Generative Modeling through ..."))
    
3. 上逆问题：用 DPS 或 DDRM 做线性/非线性重建，理解“后验 ≈ 扩散去噪 + 投影/引导”。([arXiv](https://arxiv.org/abs/2209.14687?utm_source=chatgpt.com "Diffusion Posterior Sampling for General Noisy Inverse ..."), [NeurIPS 会议记录](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf?utm_source=chatgpt.com "Denoising Diffusion Restoration Models"))
    
4. 切到 Flow：flow_matching 官方库训练 2D 分布，体会 ODE 少步优势。([GitHub](https://github.com/facebookresearch/flow_matching?utm_source=chatgpt.com "A PyTorch library for implementing flow matching ..."))
    
5. “拉直”与蒸馏：Rectified Flow + Reflow，一步或少步采样到位。([GitHub](https://github.com/gnobitab/RectifiedFlow?utm_source=chatgpt.com "Official Implementation of Rectified Flow (ICLR2023 ..."))
    
6. PINN 最小例子：DeepXDE 解 1D Burgers/Allen-Cahn，尝试残差自适应采样。([DeepXDE](https://deepxde.readthedocs.io/?utm_source=chatgpt.com "DeepXDE — DeepXDE 1.14.1.dev8+gb944422 documentation"))
    
7. FNO 上线：neuraloperator 复现实验（Darcy/NS），做跨网格泛化。([GitHub](https://github.com/neuraloperator/neuraloperator?utm_source=chatgpt.com "Learning in infinite dimension with neural operators."))
    
    

---

