


---

### 1. Current Landscape

| Method            | Projection Object   | Dimensionality Reduction | Back-Projection                                                                   | Key Features                                                                                                                                                                                                                                                                                                                                                     |
| ----------------- | ------------------- | ------------------------ | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GaLore**        | Gradient **matrix** | Low-rank SVD             | Same sub-space (fixed U, V)                                                       | Cuts optimizer memory ≈ 65 % while matching full-rank training ([arxiv.org](https://arxiv.org/abs/2403.03507?utm_source=chatgpt.com "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"))                                                                                                                                                    |
| **LDAdam**        | Gradient **matrix** | Low-rank SVD             | **Update-aware**: transforms first/second-moment stats when the sub-space changes | Keeps Adam-like dynamics with tiny memory footprint ([arxiv.org](https://arxiv.org/abs/2410.16103?utm_source=chatgpt.com "LDAdam: Adaptive Optimization from Low-Dimensional Gradient ..."), [arxiv.org](https://arxiv.org/pdf/2410.16103?utm_source=chatgpt.com "[PDF] ldadam: adaptive optimization from low- dimensional gradient statistics"))               |
| **Tensor-GaLore** | Gradient **tensor** | Tucker (multi-mode SVD)  | Per-mode orthogonal factors                                                       | Preserves tensor structure; so far limited to 4-order tensors ([arxiv.org](https://arxiv.org/abs/2501.02379?utm_source=chatgpt.com "Tensor-GaLore: Memory-Efficient Training via Gradient Tensor Decomposition"), [arxiv.org](https://arxiv.org/html/2501.02379v1?utm_source=chatgpt.com "Memory-Efficient Training via Gradient Tensor Decomposition - arXiv")) |

---

### 2. Key Pain Points

1. **Sub-space drift** – GaLore/Tensor-GaLore accumulate error if the back-projection uses factors different from the forward projection; LDAdam fixes this only for matrices.
    
2. **High-order cost** – Tucker SVD cost grows exponentially with tensor order; 4-order is already expensive.
    
3. **Error control** – No unified framework to bound error from low-rank projection + sub-space switching.
    

---

### 3. Proposed Improvement Directions

#### A. _Update-Aware_ Tensor-GaLore

- Extend LDAdam’s projection-aware updates to each Tucker mode.
    
- Maintain per-mode first/second-moment stats and apply a generalized error-feedback when factor matrices change.
    
- **Challenges:** aligning modes when Tucker ranks differ; proving equivalence between Kronecker-Tucker space and original parameter space.
    

#### B. Tucker + Nyström Hybrid

- For each mode, sample columns/rows (Nyström) to approximate its Gram matrix, then orthogonalize via Gram–Schmidt.
    
- Streaming Nyström variants can update factors online, slashing SVD cost from O(n3)O(n^{3}) to O(nr2)O(nr^{2}). ([epubs.siam.org](https://epubs.siam.org/doi/abs/10.1137/23M1585039?utm_source=chatgpt.com "Making the Nyström Method Highly Accurate for Low-Rank ..."))
    
- **Risk:** sampled columns must capture high-energy directions—use accumulated gradient energy for importance sampling.
    

#### C. Shampoo-Inspired Two-Stage Compression

- Pre-condition gradients using **Shampoo** (dimension-wise Kronecker factors) first, then apply low-rank projection within the pre-conditioned space. ([arxiv.org](https://arxiv.org/abs/1802.09568?utm_source=chatgpt.com "Shampoo: Preconditioned Stochastic Tensor Optimization"), [proceedings.mlr.press](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf?utm_source=chatgpt.com "[PDF] Shampoo: Preconditioned Stochastic Tensor Optimization"))
    
- Could give faster convergence but raises memory again; mitigate by selective layer usage or diagonal approximations.
    

---

### 4. Additional Explorations

- **Adaptive rank / temperature pruning** – grow or prune Tucker ranks based on Frobenius error or validation loss.
    
- **Integration with gradient-communication compression** – unify error-feedback buffers across memory and network bandwidth savings.
    
- **Hierarchical-Tucker or Tensor-Train** – avoid exponential cost for ≥ 6-order tensors.
    
- **Low-rank + quantization (8-bit / NF4)** – stack techniques for further memory cuts.
    
- **Incremental SVD (Oja) updates** – track slowly drifting sub-spaces without full recomputation.
    

---

### 5. Experimental & Analysis Roadmap

|Phase|Goal|Metrics|
|---|---|---|
|**Pilot**|Single tensor-FFN layer; test A/B/C individually|GPU TFLOPs, optimizer memory, val-loss|
|**Scale-up**|13 B & 70 B LLM pre-training / long-seq fine-tune|Stability (loss spikes), tokens-to-loss-X, wall-clock|
|**Ablations**|Rank schedule, Nyström sample rate, feedback window|Memory ↔ error trade-off curves|
|**Theory**|Combine LDAdam error bounds with Tucker projection; derive non-explosion guarantee|Upper-bound vs empirical gap|

---

### 6. Open Questions for the Team

1. **Memory vs Speed Budget** – What GPU RAM & throughput targets (e.g., A100 80 GB, H100)?
    
2. **Model & Data Scale** – 7 B vs 70 B, 4 k vs 32 k context?
    
3. **Primary Success Metric** – Wall-clock to reach perplexity X, or final accuracy under fixed compute?
    
4. **Integration Timeline** – Which idea (A → B → C) best fits near-term milestones?
    

---

### 7. Immediate Next Steps

1. **Reproduce Tensor-GaLore** baseline on a 4-order tensor layer.
    
2. **Prototype update-aware Tucker** (Direction A) – refactor LDAdam’s projection mapping into Kronecker form.
    
3. **Implement mode-wise Nyström sampling** (Direction B) as a drop-in factorizer; benchmark SVD runtime vs accuracy.
    
4. Meet to finalize hardware budget & choose target model for pilot experiments.
    


---

## 1 LDAdam 的“状态旋转”公式（修正 KaTeX 报错）

当第 tt 步刷新子空间时，新的秩-rr SVD 为

Gtk  ≈  UnewΣnewVnew ⁣⊤,Unew∈Rm×r,  Vnew∈Rn×r.G_{t_k}\;\approx\;U_{\text{new}}\Sigma_{\text{new}}V_{\text{new}}^{\!\top}, \qquad U_{\text{new}}\in\mathbb R^{m\times r},\;V_{\text{new}}\in\mathbb R^{n\times r}.

令 (Uold,Vold)(U_{\text{old}},V_{\text{old}}) 为刷新前的投影基，  
旧一阶／二阶矩保存在低维空间  
m~old,v~old∈Rr×r\tilde m_{\text{old}},\tilde v_{\text{old}}\in\mathbb R^{r\times r}。  
LDAdam 把它们一次性旋转到新坐标：

m~new=Unew⊤Uold  m~old  Vold⊤Vnew,v~new=Unew⊤Uold  v~old  Vold⊤Vnew.\boxed{ \begin{aligned} \tilde m_{\text{new}} &= U_{\text{new}}^{\top}U_{\text{old}}\; \tilde m_{\text{old}}\; V_{\text{old}}^{\top}V_{\text{new}},\\[4pt] \tilde v_{\text{new}} &= U_{\text{new}}^{\top}U_{\text{old}}\; \tilde v_{\text{old}}\; V_{\text{old}}^{\top}V_{\text{new}}. \end{aligned}}

随后 Adam 在新子空间继续迭代，并用

ΔWt=Unewm^tv^t+ε Vnew ⁣⊤\Delta W_t = U_{\text{new}} \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}\, V_{\text{new}}^{\!\top}

回投影到原维度，彻底避免子空间漂移 ([openreview.net](https://openreview.net/forum?id=Zkp1GuHerF&utm_source=chatgpt.com "LDAdam: Adaptive Optimization from Low-Dimensional Gradient ..."), [arxiv.org](https://arxiv.org/abs/2410.16103?utm_source=chatgpt.com "LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics"))。

---

## 2 主流三法的瓶颈一览

|方法|额外显存 / 层|每步主要 FLOPs|关键瓶颈|
|---|---|---|---|
|**GaLore**|动量  ⁣2r2\!2r^{2} + 基 (m ⁣+ ⁣n)r(m\!+\!n)r|SVD O(mnr)O(mnr)（每 kk 步）|SVD 占到训练时间 80 % 以上 ([arxiv.org](https://arxiv.org/html/2403.03507v1?utm_source=chatgpt.com "Memory-Efficient LLM Training by Gradient Low-Rank Projection"), [openreview.net](https://openreview.net/forum?id=n8MNWHfhTO&utm_source=chatgpt.com "Boosting Low-Rank Adaptation for LLMs with Cross-Head Projection"))|
|**LDAdam**|同 GaLore|SVD + 状态旋转 O(r3)O(r^{3})（可忽略）|仍受 SVD 限制|
|**Tensor-GaLore**|核心 ∏nrn\prod_{n} r_n + 因子 ∑dnrn\sum d_nr_n|每模 SVD O(dnrn2)O(d_nr_n^{2})|① 高阶张量核心爆炸；② 多模 SVD 慢 ([openreview.net](https://openreview.net/forum?id=C85eSjKenO&utm_source=chatgpt.com "Tensor-GaLore: Memory-Efficient Training via Gradient Tensor..."), [arxiv.org](https://arxiv.org/html/2501.02379v1?utm_source=chatgpt.com "Memory-Efficient Training via Gradient Tensor Decomposition - arXiv"))|

---

## 3 为什么引入 TT / 结构化 Tucker / Streaming？

|技术|如何改变成本|优势|
|---|---|---|
|**Tensor-Train (TT)**|将 NN 阶核心拆成链式 3-D cores：显存 O(Ndr2)O(Ndr^{2})，从 rNr^{N} → 线性于阶数|适合 N ⁣≥5N\!\ge 5；可变秩精细控制|
|**层次/块稀疏 Tucker**|树状或块状核心，规模 rlog⁡2Nr^{\log_2 N} 或按块稀疏计|保留 Tucker 直观结构，避免指数爆炸|
|**Streaming / Online Tucker**|随梯度流增量更新因子，不存整张量：成本 O(nnz(Gt)r)O(\mathrm{nnz}(G_t)r)|消除批量 SVD，天然嵌入 mini-batch 流水线|
|**Nyström-Tucker**|每模采样 s ⁣≪ ⁣dns\!\ll\!d_n 列近似 Gram，再 QR|将 O(dnrn2)O(d_nr_n^{2}) 降到 O(dnsrn)O(d_ns r_n)|

增量 TT（如 TT-ICE）和增量 HT-RISE 已证明在视频 / PDE 流数据上可压缩 50×50\times 且缩短 90 % 计算时间 ([arxiv.org](https://arxiv.org/abs/2211.12487?utm_source=chatgpt.com "An Incremental Tensor Train Decomposition Algorithm"), [arxiv.org](https://arxiv.org/abs/2412.16544?utm_source=chatgpt.com "[2412.16544] Incremental Hierarchical Tucker Decomposition - arXiv"))。

---

## 4 在 LLM 微调中，高阶梯度是否必要？

- **典型 Transformer 权重**：绝大多数是形如 Rdout×din\mathbb R^{d_{\text{out}}\times d_{\text{in}}} 的矩阵（QKV、MLP-FC）。  
    把它们强行 reshape 成高阶张量并不会带来天然“多模相关性”，低秩矩阵方法已足够逼近梯度能量；  
    额外维度只会引入 **指数级核心** 与复杂 SVD，得不偿失。
    
- **可能的例外**
    
    - **多头注意力**：若把 (head,dh,din)(\text{head},d_h,d_{\text{in}}) 视作 3-阶，可利用 head-wise 低秩；
        
    - **MoE / Jagged Block**：张量形状映射天然存在 block 结构，可用块稀疏 Tucker；
        
    - **科学算子学习 (FNO, U-Net 3D)**：权重本就带空间×\times通道×\times方向多模结构，高阶降秩能大幅节省显存——这正是 Tensor-GaLore 的目标场景。
        

结论：在主流 LLM 微调里，**矩阵 GaLore + LDAdam** 已是性价比最高路径；只有当模型或任务本身呈现“真 N-阶”结构时，Tensor-GaLore 及其后续才显优势。

---

## 5 更复杂分解的快算法：挑战与解法

|痛点|具体挑战|可行解决思路|
|---|---|---|
|**迭代稳定性**|TT/HT 每步需多次 QR/SVD，误差累积 → 核心漂移|_正交化缓存_：保持 TT-cores 左 / 右正交；周期性重新正交|
|**秩自适应**|训练早期秩不足，后期秩冗余|_增量秩扩张 + 热度剪枝_（TT-ICE，HT-RISE）自动加／剪核向量 ([epubs.siam.org](https://epubs.siam.org/doi/10.1137/22M1537734?utm_source=chatgpt.com "An Incremental Tensor Train Decomposition Algorithm"))|
|**GPU 吞吐**|TT-SVD 中的 unfold-SVD 带内存搬移，GPU 利用率低|1) **Fused TSQR + matmul** 内核 ([arxiv.org](https://arxiv.org/abs/2102.00104?utm_source=chatgpt.com "Performance of the low-rank tensor-train SVD (TT-SVD) for large dense tensors on modern multi-core CPUs"))；2) 随机化 TT-SVD，只读一次数据|
|**异步流水**|Streaming 场景下梯度 → 分解 → Adam 必须重叠|_细粒度流式 API_：切分梯度通道 / head；一边计算下一 mini-batch，一边增量分解|
|**误差控制**|随机 / 增量方法需明确界限|引入 **误差反馈 (EF)**：保存低秩残差累积项，仿照 LDAdam 延续收敛性证明|

---

## 6 建议落地路线

1. **LLM 继续用 LDAdam**：先解决 SVD 加速（随机化或 Nyström），再考虑 per-head/块结构。
    
2. **科学算子学习**：在 Tensor-GaLore 上实验 _Streaming Tucker → TT-ICE_；并加入 LD-style 误差反馈，验证 6-阶以上 PDE。
    
3. **通用库**：封装 TSQR-based TT-SVD、增量 QR、Nyström Tucker 为可插拔 `torch.autograd.Function`，以便在 GPU 上调用。
    

如需推导更具体公式或代码原型，请告诉我！