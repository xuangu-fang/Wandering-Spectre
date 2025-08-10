> work with PQ

- streaming/online diffusion : 将隐空间（Tucker core）的 GP 假设 + function post. sampling 做成 Bayesian filter 的形式，设计好 online “平滑滤波”/ 梯度融合的形式和算法，使得 diffusion 对于变化的观测值，能够在隐空间的 cross-frame 地 中做实时的调整 （比如 第一帧/ X 帧的观测值变了（PDE 初始值/ 天气预报观测到极端值  ） ），后面帧的生成过程也应该在 training-free 的情况下 实时变化 
	- Generative KF/Smoother


问题：
- 主流的视频/cv任务中, 直接就把 action/label 编码进了预训练的 data pair 中，所以现在主流的视频模型都是 cfg， 不适用 “平滑滤波”适配的 cg / DPS 式的条件生产
 - 什么样的任务(视频？具身？物理场？) 更加适配 cg / DPS 式的条件生产？ 
			- 没有 simulator， 只能从真实世界中提取的 traj 数据
			- 有simulator，但是对应的 likelihood / reward 稀疏/带噪/ costy
			- w/o simulator,  traj 没有显式的/well-defined 的 action / label 来计算llk
			- 后续的 DPS / cg 可以怎么更加玩出花样来？

