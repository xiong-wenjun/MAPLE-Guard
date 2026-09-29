# AgentSafe、INFA-Guard 与 GUARDIAN 的公开版本复现

本轮将“公开源码检测器复现”“按公开流程重新训练”“接入 MAPLE 的任务适配”分别记录。任何一项通过单元测试，都不代表正式 benchmark 已完成。所有源码、模型、训练输入与运行产物保存在服务器 `/mnt/public/data/wj`，不在本地 clone 仓库。

## 已落实与仍需取得的内容

| 方法 | 本轮落实 | 严格表述与剩余条件 |
| --- | --- | --- |
| AgentSafe | 固定官方源码；直接调用官方关系生成函数；核验 15 个 seed 的确定性和双向对称关系；保留已有 HierarCache 组件 | 关系生成机制可复现，原随机种子未知。公开 demo 没有论文的 cosine criteria 准入机制，不能据此恢复完整论文配置 |
| 原生 INFA-Guard | 固定源码和有效训练参数；取得官方 CSQA train；排查五个 held-out 文件；建立隔离执行器、逐轮标签校验和原生权重严格加载检查 | 必须实际完成训练后才有可用双头权重。Qwen 替换原生成模型时明确标为独立重训与模型替换 |
| GUARDIAN | 固定官方源码、BERT 权重；增加 released_detector 协议；真实静态/时序检测器差分检查通过 | 无需预训练检测器 checkpoint。检测器每轮在线训练。宿主 prompts、轮数、通信调度和投票仍是 MAPLE 适配 |

配置锁：`configs/baselines/official_reproduction.lock.json`。它包含审计源码文件 SHA-256、上游 commit、BERT 与 CSQA revision，以及实际生效的超参数。执行时哈希不匹配会报错，不会静默替换。

## 1. AgentSafe：恢复能恢复的官方配置，明确未公开的参数

固定 [AgentSafe cc253ad](https://github.com/junyuanM/Agentsafe/tree/cc253ad48532fa6614a27557587086cfb87968ed)。

`Code/initial.py::initialize_relations` 对每一对身份均匀抽取 Family/Friend/Colleague/Stranger，再写入双向关系。因此缺少 relations.txt 不代表生成流程缺失。新工具仅抽取并调用这个官方函数，不执行 initial.py 顶层的模型调用。生成时声明 seed，固定身份列表与顺序；MAPLE 数字 agent ID 的映射是宿主适配，self_level=4 也是宿主约定。

```bash
python tools/official_reproduction.py agentsafe-relations \
  --source /mnt/public/data/wj/baseline-references/AgentSafe \
  --names Agent_0 Agent_1 Agent_2 --seed 42 \
  --output-dir /mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/new-seed42
```

输出为官方格式 relations.txt、MAPLE policy JSON 和来源记录。工具**不生成假定为官方的 criteria 或阈值**。

[论文公式 8–9](https://arxiv.org/html/2503.04392v2) 使用 criteria 相似度与阈值；在本轮审计的公开版本中未找到对应 criterion 文本、准入阈值、完整 embedding 配置与原实验关系实例。process.py 的 0.2 是攻击结果评估相似度，不是准入阈值，不能挪用。

公开 demo 的反思逐行删除不合理内容；已有 agentsafe_full 的反思以完整记录为单位，且支持 junk 状态。因此两者不能无说明地都叫原样复现。推荐分别保留：

- **released demo**：运行官方社交身份场景、四级 memory、官方提示语和逐行反思，用来验证公开代码。
- **paper-components / MAPLE adaptation**：已有完整生命周期组件，明确披露任务 policy、criteria、embedding、阈值和反思粒度，禁止宣称原实验配置已恢复。

要关闭“完整论文配置”缺口，应取得作者的关系/身份初始化文件、criteria、准入阈值、相似度编码器和归一化规则、反思频率及记忆初始化内容，并记录资产哈希。可向作者提出这份明确的资产清单；本轮没有代发邮件。若只能自行设定，应在独立开发集预注册规则与误杀约束后选择阈值，测试集不得用于选择，论文必须标为适配版。

## 2. INFA-Guard：严格重训路径

固定 [INFA-Guard 80b1156](https://github.com/yjzscode/INFA-Guard/tree/80b1156cb22d576d9149046c36c540c728a43dde)。

当前两个 G-Safeguard 权重的 out.weight=[1,1024] 对应单输出网络。原生 MyGAT(guard=ours) 有共享层、四个时间分支及每个分支的 malicious/infection 双头，不能靠改输出尺寸、补随机头或重命名恢复缺失的训练。

公开脚本的有效规格：

| 项目 | 固定配置 |
| --- | --- |
| 训练数据 | CSQA train；原脚本生成模型 gpt-4o-mini |
| 图 | 8 nodes；sparsity=0.2/0.4/0.6/0.8/1.0；attackers=1/2/3/4 |
| 采样 | 每格生成 20 张候选图、取 40 条对话；20 格共 800 条 |
| 对话轮数 | 初始生成 1 次 + num_dialogue_turns=3，再生成 3 次，共 4 轮 |
| 编码器 | all-MiniLM-L6-v2，384 维；不替换为宿主 Qwen embedding |
| 模型 | hidden=1024、heads=8、layers=2、dropout=0.2；四时间分支 |
| 训练 | 50 epochs、batch=32、lr=0.001、Adam weight_decay=0.0002 |
| 调度与损失 | CosineAnnealingLR(T_max=10, eta_min=1e-5)，alpha=1、beta=0.5，selective_training |
| 模型选择 | malicious 与 infection 验证准确率之和 |
| 原发布划分 | 前 80% 行训练、后 20% 行验证；另报告题目层面的 train/val 重叠 |

保留并披露公开代码的细节：edge_attr 取目标节点回复；训练初始 scatter 使用源节点，评测使用目标节点；trainer 的 validation 没有实际传入 use_branch_aggregation=True；dropout 命令行参数未传入构造器，但构造器默认恰为 0.2。不得悄悄修正这些细节后仍声称执行同一公开版本。

已取得 [CSQA 官方训练资产](https://huggingface.co/datasets/tau/commonsense_qa/tree/94630fe30dad47192a8546eb75f094926d47e155)，9,741 行，校验 SHA-256。与用户五个测试文件按题干、大小写及空白归一化比较，精确重叠为 0；这不是语义去重结论。输入与测试文件 SHA-256 写入训练目录 provenance.json。

新服务器可运行 `official_reproduction.py prepare-infa-data` 重建同样审计；需要 pandas 和 pyarrow。本服务器 pyarrow 安装在独立 `/mnt/public/data/wj/baseline-training-runtime`，使用 PYTHONPATH，不修改已有实验依赖。

执行器 `tools/run_infa_release_stage.py` 的 stage 顺序是 prepare → generate（grid-index 0..19）→ merge → embed → train。命令清单由 `official_reproduction.py infa-recipe` 生成，原模型和 Qwen 替换版分开保存。原生 evaluate/utils/train 模块在专用进程内重新导入，防止误调用 MAPLE 同名包。执行器拒绝修改后的 recipe、源码、模型资产、测试文件或训练输入，API 密钥仅注入环境。历史 `released` 配置的 API 审计不修改请求参数，但附加了拒绝截断与空回复的门禁；这比官方发布代码的正文消费规则更严格。该历史行为保留，新增的官方预算配置见下文。

```bash
# 下列路径均是服务器路径。prepare 只允许新工作目录。
python tools/official_reproduction.py infa-recipe \
  --source /mnt/public/data/wj/baseline-references/INFA-Guard \
  --job-dir /mnt/public/data/wj/baseline-reproduction-20260929/infa-original-new \
  --dataset-dir /mnt/public/data/wj/baseline-training-inputs/csqa-train \
  --model gpt-4o-mini --seed 42 \
  --output /mnt/public/data/wj/baseline-reproduction-20260929/infa-original-new.json
```

`--protocol-check` 只生成两条流程检查对话，写入独立 protocol-check 目录；无法作为正式训练阶段输入。合并前要求 20 个文件、每文件 40 条、总计 800 条，每条必须包含 8 个节点的 4 轮非空回复和真实逐轮 infection 标签。拒绝用最后一轮感染结果补齐缺失标签。合并前记录文件顺序，拒绝重复合并导致 dataset.json 被自身纳入。

真实流程检查发现：Qwen 使用官方 1024 token 上限、默认思考时，首批 8 次请求有 5 次截断，其中 4 次正文为空。因此保留原参数失败记录，并新增显式 qwen_no_thinking 生成配置，仅增加 enable_thinking=false 请求参数，仍保留 1024 token、官方 prompts 与训练参数。它是 Qwen 模型替换协议，不是原 GPT-4o-mini 配置；不能将其失败或不完整记录混入正式训练集。

每个已完成阶段记录实际依赖版本。当前验证环境与上游 requirements 的所有版本并非完全一致，因此不声称复原原作者的二进制环境；原代码与适配入口的差分测试在同一记录环境执行。

训练完成后仍需 `official_reproduction.py validate-infa-checkpoint`：安全加载、全部参数 strict=True、真实 forward 输出 [nodes,2]。结构合格不代表确实训练，必须同时保留训练日志、数据来源和 checkpoint 选择记录。随后再运行 Qwen/Gemma 评测检查。**本轮尚未产出训练完成的原生 INFA 权重。**

## 3. GUARDIAN：官方检测器已可验证

固定 [GUARDIAN 60d7b8b](https://github.com/JialongZhou666/GUARDIAN/tree/60d7b8b6a9e64b87134b1078bfbba0f1e7b5b50b)。明确标为 JialongZhou666/GUARDIAN 的时序图方法，与 Inspector 使用不同方法 ID；新结果不继承旧论文行的身份歧义，旧结果需要另行核对。

GUARDIAN 在任务/轮次中重新训练检测器，所以“缺预训练检测器 checkpoint”不是阻塞。必需资产是官方代码和 [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)。

新 released_detector profile：

- 按官方 parse_single_choice 获取节点答案；活动节点组成无自环的完整有向图。
- 首轮静态 DOMINANT：hidden=128、2 层；后续 temporal GIB：hidden=64、2 层 GNN、2 层 Transformer、4 heads、beta=0.001。
- feature_weight=0.3，Adam lr=0.001，每轮重新训练 20 epochs。
- CPU 运行，保持官方 BERT 冻结/可训练模式、dropout 和 temporal GIB 随机性。
- 每轮移除最大异常分数节点，不增加私设阈值；活动节点映射到实际宿主 agent ID。
- 每次运行记录源码与 BERT 哈希；不允许变更 epoch 或悄悄退回其他编码器。

```bash
python tools/verify_guardian_release.py \
  --source /mnt/public/data/wj/baseline-references/GUARDIAN-60d7b8b \
  --bert-dir /mnt/public/data/wj/baseline-references/bert-base-uncased-86b5e093/bert-base-uncased \
  --output /mnt/public/data/wj/baseline-reproduction-20260929/guardian/new-parity.json
```

已完成真实 BERT 对照：固定 seed=42 的三节点静态/两轮时序输入，适配入口与直接官方入口均使用 20 epochs，图、节点文本、移除索引一致，分数最大绝对差为 0。这是限定输入下的检测器差分验证，不是所有输入的形式证明，也不是 benchmark 成绩。

评测参数新增 `--official-defense-guardian-profile released_detector`、`--official-defense-guardian-bert-dir` 和 `--official-defense-guardian-epochs 20`。原 host_graph 模式保留，已运行的快照与队列不修改。新矩阵显式要求源码和 BERT 路径。

仍需披露宿主适配：MAPLE 的提示语、固定轮数、通信拓扑、memory、最终投票与官方 agent framework 不同；AppWorld 当前仍是动作选择 proxy。公开检测器输入图为完整图，不代表宿主通信图被改为完整图。4 拓扑对比应说明这一方法内固定预处理。

## 实验准入和主表设计

修复后的方法先通过源码/资产检查和组件对照，再进行 Qwen/Gemma 各两题完整协议检查；只有生成、检测、过滤与记忆事件真实参与且无 API/依赖错误，才进入新的正式 200 题、15 seeds × 4 拓扑矩阵。运行目录、模型名、协议、源码版本必须与旧适配实验区分。

主表写清方法身份与实现来源。AgentSafe paper-components、Qwen 重训 INFA、GUARDIAN released detector in MAPLE 均应在组件表列出宿主改动；原公开框架的 native 复现实验作为附录核验，不与当前 AppWorld proxy 成绩混成同一口径。不得把协议检查题数、单元测试或随机初始化模型当正式结果。


## Declared Qwen recovery and automatic test handoff (2026-09-29)

The first 1024-token Qwen generation grid failed after 191 complete replies and one truncated reply; upstream saves its dataset only after all 40 dialogues finish. Those historical replies were not retained and cannot be reconstructed or counted as training data.

Use the separate `qwen_no_thinking_recover` recipe in a fresh workspace. It leaves the pinned official source, graph sampler, infection labels, four turns, 800-dialogue grid, MiniLM encoder and 50-epoch optimizer recipe unchanged. The explicitly non-author runtime adaptation adds:

- Atomic per-request records for accepted nonempty `stop` replies, indexed by deterministic invocation ordinal and original request hash; restart replays completed requests and rejects prompt/cache drift.
- Token budgets 1024, 2048, 4096, 8192, used in that order only while the same request remains incomplete; no prompt rewrite and no incomplete reply admitted to training.
- At most three wrapper attempts on connection/timeout/429/server errors per token budget; other API errors fail immediately.
- A complete 64-response journal for two protocol dialogues or 1280 responses for each 40-dialogue training grid, with an aggregate integrity hash.
- A bounded controller: two protocol dialogues, 20 grids, merge, feature generation, 50 full epochs, native checkpoint validation, two-backbone smoke checks, then AppWorld-200 evaluations. Failed prerequisites block subsequent stages.

Run `tools/run_infa_training_pipeline.py --help` for the controller arguments. Run it from a frozen repository snapshot. Credentials, response journals, checkpoints, calibration data and all benchmark outputs remain under the authorized server directory outside Git. The selected checkpoint is independently retrained; it is not an author checkpoint. Qwen generation and token-budget recovery must be disclosed when reporting this baseline.

Validation includes restart parity using the actual pinned upstream graph generator and synthetic test responses: interrupt after the 36th request, retain 39 complete concurrent replies, replay them and make 25 new calls. The reconstructed two-dialogue graph, replies and per-turn infection labels exactly match the uninterrupted fixture. This is an integration test, not training or benchmark data.


## Qwen 官方预算配置：训练生成与评测分开处理

`qwen_no_thinking_released_budget` 使用官方生成器的固定 `max_tokens=1024` 和正文消费方式。证据是固定版本的 [MAS/agents.py:62–72](https://github.com/yjzscode/INFA-Guard/blob/80b1156cb22d576d9149046c36c540c728a43dde/MAS/agents.py#L62-L72)：上游直接返回 `choices[0].message.content`，不检查 `finish_reason`。此前额外要求所有回复以 `stop` 结束、并把预算升到 8192，是 MAPLE 完整性恢复适配，不能当成原作者的生成协议。

新配置只用于训练生成：

- 请求预算始终为 1024，正文非空且结束原因为 `stop` 或 `length` 时，原样交给官方生成器；不续写、不补全、不改提示或答案标签。`length` 只表示消耗了官方预算，不被描述为自然结束的完整答案。
- 空正文、其他结束原因及最终 API 错误仍失败。连接、超时、429/服务端错误保持最多 3 次 transport 尝试；没有 token 阶梯。
- 原子 journal 保留整个 API response、原始 `finish_reason`、请求/回复哈希与 accepted token budget；阶段报告增加 `accepted_finish_reasons` 次数。64/1280 次回复、四轮八节点、800 条训练对话、50 epochs、源码/数据/权重校验保持。
- Qwen 替换 GPT-4o-mini、关闭思考和缓存/网络恢复仍需披露。旧 `qwen_no_thinking_recover` 的拒绝截断及 1024→8192 行为保持，旧运行与冻结源码不修改。
- **评测仍使用 strict 协议拒绝截断。** 接受训练生成的官方封顶正文不降低 Qwen/Gemma smoke 或 200 题评测的成功门槛。

用 `official_reproduction.py infa-recipe --generation-profile qwen_no_thinking_released_budget` 为全新工作目录准备 recipe，正常执行 prepare，再由控制器运行。不得把旧工作目录的完成 marker 改名作为新配置结果。

如复用旧 journal，仅允许从 ordinal 0 开始连续的非空 `stop`、`accepted_max_tokens=1024` 前缀；到第一个缺失、扩容或其他结束原因即停止。后续 1024 回复也不能直接复制，因为它们可能依赖此前不同的上下文。`tools.infa_generation_recovery.import_released_prefix(source,destination)` 要求全新目标目录，保存来源目录、每个原文件 SHA-256、整个前缀 SHA-256，以及停止原因；重放时仍逐条核对新请求哈希。来源元数据保存在 journal 的 `import-provenance/provenance.json` 子目录，阶段完整性报告记录其哈希。

只读诊断证据：旧运行 grid01 ordinal66 的请求 SHA-256 为 `d6ee6b447139969cad43d27549ce4638be041384ec0091230a508c0826eb73d0`，对应第三条对话首轮攻击者 agent2，提示要求为指定错误选项辩护。1024、2048、4096 阶梯均返回 `length`；截断正文未保存，不能断言其具体内容。grid00 ordinal71 在 1024 返回 `length`，2048 重试却以 126 个 completion tokens 返回 `stop`，表明再次请求不能假定逐 token 确定。诊断时可复用前缀为协议 64 条、grid00 71 条、grid01 66 条；实际导入必须重新核验当时磁盘内容。
