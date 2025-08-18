

方法论输入：
1. GP-SSM - new inference 方法？（VI-based）以及 拓展到高维（check factor-GP-SDE paper：https://github.com/SeyoungKimLab/FactorialSDE）
2. latent dynamic model - 其他建模方法 :
       - GNflowNet -ask dinghuai's talk?
       -  deep state space，
       -   neural process/flow 
       -   Stein partical filter?
3. latent diffusion - how and where?
       - check research idea sheet for detail, indexed-feature + index-aware attention + observed values in patch -> as the input of the conditional diffusion for non-linear tensor  
4. (online) inference of other sparse prior (beyond Spike & Slab)
   - 与程磊聊过，意义一般
5. fusion/kernel model over graph / hypergraph
6. 一些 ai4physical 的相关问题定义 与解决范式 （check collection in list）

输出：
1. 知乎文章：补全介绍 GP-SS 推导的 文章，从Wiener-Khinchin theorem 到 构建（顺带介绍BCTT,以及时序动态张量）
2. 知乎文章/latex 文档，详细介绍 GP-SS后续拓展，以及其他一些latant dynamic 思路
3. DEMO: 将 talin PDE (NV-equation ) 视作时序张量，并用GP-SS 尝试效果
4. DEMO: online SS infernce for CP (tensor rank selection)


Big View:
1. tensor <-> hypergraph
2. diffusion <-> latent  structure / tensor

-----------------------------------------------------------------------------------------

2024/10/10:

AR diffusion and diffussion forcingf的方法 + LOB image token 的思路，能否用在讲图片/high-dim tensor 分解得到的 functional factor 上？

2024/3/6:
领域判别器 + 强化学习 + diffusion + 小正样本数据？


2024/2/22:

DA 数据同化是一个大坑，目前的FunBAT 的性能和上限都不够好，需要新的思路和方法。但如果能做好，在气象、物理场和复杂系统领域有相当的应用前景。(https://github.com/nansencenter/DAPPER?tab=readme-ov-file)

10/26 update:
从唐志军朋友圈得到的灵感：function tensor train 可以解 PDE，该思路可以拓展到 latent dynamic based tensor


10/20 update:
学习 等变DNN时的一些idea：
1. 将等变的性质嵌入到tensor中
2. 将时空tensor 变换为极坐标，实现完整的functional tensor framework, 以及将周期性质嵌入到对角度的连续建模里。对于距离r, 甚至可以定义多种读论，实现对multi-task 的建模。 

10/8 update:
后ICLR时代，主要精力集中在 整理之前工作 + 开拓新思路 + 对之前的一些idea 做可行性验证 proof of concept, 为后续的合作和新阶段积累弹药。大致的任务可以分成以下几类：

1. TIF-down： 借用天气数据，对气象建模 + scalable latent dynamics 建模/推理的新方式，同时保持和DAMO 的联系（详情见TIF-down 项目的log 与总结， 以及GP-SSM 的探索思路） 
    - low-rank functional tensor的思路走不通，需要新的思路 
   
2.  Tensor + PDE : 在做 FunBat 过程中，发现貌似有文献 利用 TT 解 PDE , 建议推进 & check， 看能否和FNO， PINN 等联系起来 （possible connection: tailin wu）

3. Tensor + attention for multi-task/inner structure: 同样与TIF-down 联系起来，做向量场、张量场的建模
   

4. Tensor + quantum (quantum ML): tensor 结构与量子力学的张量网络联系起来，check 相关github repo / survey from 徐增林（https://arxiv.org/abs/2302.09019）

5. 大模型
   - 压缩/量化 idea from xuyin 
     - cascasde TT / 对res data 持续做 tensor learning, 似乎能有不错效果 （貌似等价于不断增加CP的R，但是分阶段训练）
     - 能否应用到其他领域中？取得效果提升？
        
  -  in-context learning + TT (general low-rank representation/ world model embedding?)  - 另一种实现multi-task TT的思路

6. 开源包/知乎推广的整理


9/11 update:
1. 在TIF-down 项目的探索中，明确了：
   - 直接用 tensor 对 dense structural dynamic 建模效果不够好
       - 需探索更有效的、结合NN module 的建模方式
   - SS-GP + CEP 不够 scalable，特别是对大尺度、百万以上的天气数据  
       - SS-GP + Stein filter? /SMC
       - SS-GP + SVI 
       - sparse GP / neural process (gradient-based update) 
       - biger question: 怎么结合 NN-based 的 显空间建模 以及 tensor-based的隐空间建模

2. 以 Lorenz 96 sysytem 为代表的混沌系统，似有大量的金矿可以挖
   https://chat.openai.com/share/7088f903-4169-494a-b1d3-b63d895f18b8

3. tensor + hypergraph, feature-based lapalacian -> 详见 hyper-graph GCN (https://arxiv.org/abs/1809.02589)

4. multi-task/aspect tensor -> attention ?

06/19 update: 
1. 放弃 ICDM ， 争取ICLR *3 (BayTIDE + 时空数据超分 + BaconFat 加实验)
2. 探索制定 AI4Sci 方面的战略和执行计划（能做的项目/能找的合作/可能的创业or开源项目）
       - check 集智读书会内容
3. 探索大模型LLM可能和当前研究结合的点。掌握基本的技能
       - play with HuggingFace + GPT2 + TS demo,
       - 关注用LLM 联动下游小模型的一些前沿文章+demo


2023 summer (in Ali)

project:

BayTIDE：online learning 已取得不错效果，后续有待进一步优化。
- for irregulate time-step? - 加到附录中
- for frocasting ? - 长期效果不加。考虑加到附录中
- offline version? pallara update msg over all time step?

Tensor 4 PDE: check tailin's paper and make plan 

Online tensor rank selection-> read chenglei's paper，scedule meeting? 

BaconFun 项目：阅读 两篇确实相关文献，构思加实验？新的方
- 能否直接拓展到 应用于spatial-temporal data 投ICRL？）
- 除了GP-SS + CEP， 探索其他能对 Bayesian latent dynamic 建模的 新方法 

diffusion <-> latent  structure / tensor -> check latent space diffusion 新进展


