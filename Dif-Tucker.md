


未来idea collection:

- streaming diffusion : 将隐空间（Tucker core）的 GP 假设 + function post. sampling 做成 Bayesian filter 的形式，设计好 online “平滑滤波”/ 梯度融合的形式和算法，使得 diffusion 对于变化的观测值，能够在隐空间的 cross-frame 地 中做实时的调整 （比如 第一帧/ X 帧的观测值变了（PDE 初始值/ 天气预报观测到极端值  ） ），后面帧的生成过程也应该在 training-free 的情况下 实时变化 
- Generative KF/Smoother