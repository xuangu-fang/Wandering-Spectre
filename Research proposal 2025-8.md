## 1. 生成式物理信号建模类 

- Generative Filter (seq. diffusion):
	- 把SDIFT 进行拓展，把bayes 的序列建模 和 guided 生成结合起来
	- 亮点：尝试把 guidance 内部的“结构” 给用上，follow 
	- 问题：适合做什么数据和任务？


- AR-based structure dynamic:
	- 沿着SY现在的逻辑，把AR 做多元时序、张量时序去打通
	- 探索多模态、控制、和编辑 信号如何更好地加入
	- **小圣杯1**：如何在结构数据的AR中去确定order？
	- **小圣杯2**：如何在结构数据的AR中去加入多模态的控制和编辑信号？


- 水声场景：
	- 现有问题：
		- 高频泛化刻画，
		- latent force, 
		- 带不确定性的 边界条件（海底地形参数），反问题infer
	- 这些经典问题如何用生成模型建模
	- 如何把Physical constrain 更加优雅地加入


- Generative model in Freq. Domain + low-rank:
	- 短时傅里叶变换得到的矩阵有 low-rank 性质吗
	- 频域中的特征能做类似 

## 2. 物理信号表征类（low-rank tensor as infra）:

> 目标：把tensor 做成物理信号表征的新基建，也做成个人的品牌

- 验证 functional tensor 做大规模、跨任务预训练表征器的可行性
	- follow PDE-transformer / Bert 的思路
	- follow graph-Foundation model 的思路

- 尝试建立 类似 CLIP 之类、在声光电磁等信号场的多模态基建工作
	- follow astro-clip 的思路，思考和定义 水声/电磁 领域的 “多模态”是什么？
	- 如何为物理信号的多模态建立通过的表征


- 对张量模型本身的进一步泛化和探索：
	- 球面张量分解
	- 对向量场、张量场（电磁场）的表征
	- 解耦、低秩表征，怎么和 Diffusion， AR 更好的结合起来


$$\begin{aligned}$
\mathbf{T}(x^\mu) &= \mathcal{G} \times_1 \mathbf{f}_1(x) \times_2 \mathbf{f}_2(y) \times_3 \mathbf{f}_3(z) \times_4 \mathbf{f}_4(t) \\
&\downarrow \text{物理约束层} \\
\text{s.t.} \quad & \partial_\mu T^{\mu\nu} = \mu_0 J^\nu
$\end{aligned}$$

## 3. Agent 

- **圣杯1**：auto-benchmark + auto-codebase  

- **圣杯2：基于sensor 的 时空多智能体网络

- 小**圣杯**：通过agentic 的方式，把经典任务和模型泛化性提升一个维度 （类似）

- 目前的实践1：RD agent

- 目前的实践2：Battery-Agent-with XF
	- 把agent 当做 black-box 优化器，去求解反问题
	- “solver / simulator in the loop”
	- 未来可以拓展到更广泛的 inverse design 任务中 


## 4. Large Signal Model  
- **圣杯：** wave2vector、signal2vector 的 foundation model
	- 可能是通过RMKV之类的 RNN 训练得来？


## 5. Others
- low-rank grad? -- 和 Beheshtch 持续合作
- 天气-气象
- 社会信号/计算法学 -- june
- Finance