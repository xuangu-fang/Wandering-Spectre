
-weizhen Qi
- 署名-deepseekai
-

- V1: 单纯复现不同的scaling-law:
- V2: 非llama 架构--DeepSeekMOE + MLA
	- 极致的 KV-cache: save 93% 
	- MOE from Mistra: 8 x 7B
		- 更多的专家：shared + Routed
		- 更复杂的 router
	- EL-attention / MLA: 
		- gap: 推理bottle-neck 在显存IO
		- solution: 用 latent-state h 替代 多头KV
	- 
- R1: 
	- 验证PRM, MSTC 没大用
	- ProphetNet / MTP: 
		- gap: 
			- 预训练语料利用效率不足- tr/ar本身决定的？
		- solution:
			- 多个词预测-提高远期规划
			- prefix prompts <-> adapter (1,2,3, xxx, 4,5,6 ) —— xxx 是拟合空间（额外的参数），4，5，6 是强化学习监督信号，让模型自行搜索xxx, 而不是限制他怎么思考/给他
				- agentic workflow 长期来看是限制！



- 组织方式：
	- openai-更加灵活频繁的绩效评价：扫描最前沿的技术并快速复现


### R1 复现：
 - Dense: 4 台A800, 386h, 10w 元 （72B-Qwen）：
	 - stage1: SFT 2h
	 - 
 - V3 成本： 560w刀 
 - V3 + R1: 560w刀 + 5w 刀左右

Deepseek-R1-Zero 训练：pre-training + 直接 RL (跳过依赖标注的post-training: )-甚至PPO/DPO都可以，不依赖GRPO

R1： V3->冷启动SFT(几千长COT)->面向推理的RL (准确性和格式)-> 全量SFT (60w推理+20w通用，等效V3 post-training)->全场RL(无害、偏好)


蒸馏版：用R1 推理样本 SFT + qwen-xB-(chat-instrau),将R1推理能力蒸馏到小模型中 
- 成本：32B 2台8卡A800 4 天


杨耀东-ai 对齐：RL 和 post-training
- RLHF使得1.5B 超越了175B
- 1% RLHF->1% 预训练

- RLHF/DPO 对于100B以上不适用 （完整的有很耗显存）

- 胡克定律<->大模型弹性（抗拒对齐）

- 树结构奖励有用？无用？

- VLA/多模态的对齐，agent的对齐？  Vidoe-phy [[physical-world-signal AI]]
- RL-PF 对物理世界和规律的对齐

- 所有的模态token在隐空间是一致的-柏拉图假设
	- 广义张量分解？？？跨越模态的张量
	- media-fundation


- AI 成本的降低：1年10倍，但是很多trick 很难沿用到下一代 （resnet->bert->gpt->deepseek）, 因为每一代的核心矛盾不一样。下一代GPT5，集成了sora(物理理解), o1(推理), llm(常识), 4o(多模态)

- 下一个阶段是什么：
	- post-training boosting
	- test-time ? 
	- 多模态（爬取，人标、online PPO, RL-合成数据，蒸馏数据）- 但用不好
	- agent-爆发前夜-
		- what's beyond RGA + function calling? 
		- 自动生成的workflow
	- 新的scaling-law: in multi-agent? in first-principle? 

- 开源：团结全球开源社区
- 优势：产品力+行业场景

- RL-fine tunning: 没有清晰reward 的领域，怎么做RL
	- 数据飞轮，场景变现的新机会


## 圆桌

kimi 和 ds:
- 思考类似：length counts: 文本过长>靠分块，但分块本身是w.o参数

文本&多模态：
- 信息密度的高低天然区别
- 文本无损、图像有损，完备性的区别
- 通用性：文本输入输出天生如此，但多模态不一定
- 3d 物理场景的tokenize 会是3dGS 吗

无边界探索，ds? 国家队？msr？

奇点：ai的自指和自我提升？

新渠道：aigc 是出海渠道、造工具的必经之路

### 问题：

多模态的tokenize: 3dgs/图形学/

agent-应用和商用之外的non-trival 

ai4science：diffusion , 第一性原理, 

生产关系-内容生产者的盈利，开源--专利制度的退潮，人的价值何在?

