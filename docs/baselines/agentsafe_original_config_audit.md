# AgentSafe 原配置审计与 AppWorld 候选

核查日期：2026-09-29。完整原论文配置不能从已检查公开资产恢复。用户已接受明示的完整组件适配。独立开发校准已完成，15 个官方生成器关系实例已准备；正式评测由主编排在协议检查后启动。

## 公开来源

已检查 [arXiv v1](https://arxiv.org/html/2503.04392v1)、[v2](https://arxiv.org/html/2503.04392v2)、两版 TeX 附件和 [官方仓库](https://github.com/junyuanM/Agentsafe/tree/cc253ad48532fa6614a27557587086cfb87968ed)。GitHub 只有 main，没有 releases/tags/notebooks；13 次提交的全部文件树和 10 个不同 Python 源码版本均已检查。附件是论文 TeX、样式、引用和图像，没有实验实现或参数文件。

公开主表为 RIOH/WCEI、6 个普通 agent 加 1 个 attacker、10 轮。Eq.8–9 定义“每个 criterion cosine 严格大于 theta”，但未给完整 criterion、theta 或准入编码器。完整身份抽取/匹配提示语、原随机关系实例和完整推理参数也未恢复。

v2 在 Eq.3 和 Algorithm 1 把 v1 的发送者/接收者权限比较改为“消息安全级别不高于接收者权限”。Algorithm 1 的源文件括号把检测置于每轮循环内，支持按 review_interval=1 实施；附录 H 的 5/10/15/20/25/30 是报告交互轮数，不能当检测间隔。表格实际运行的独立数值周期未披露。

process.py 的 text-embedding-ada-002 和 0.2 用于攻击结果评价，不在 agents.py 的准入链路。正文约 0.7 是报告相似度结果，也不是准入 theta。

## 文件与审计证据

- configs/baselines/agentsafe_original_configuration.audit.json：官方可恢复字段、未知字段、SHA-256、历史核查与证据路径。
- configs/baselines/agentsafe_appworld_candidate.json：完整可审阅候选；official_configuration_recovered=false；校准完成后 runnable=true，theta=0.18154229499203256。
- configs/baselines/agentsafe_appworld_candidate.criteria.json：三条独立编写的候选 criterion，非作者发布资产。

原始公开响应和论文附件保存在服务器 /mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/source-audit-20260929/。v1 tar SHA-256 为 5251a33fa61c60b323d8b947fa17a34c7e42a9d377986a09f398fa85eb604517，v2 为 2b6a153856b5a2001adf14fc21f8e292e6f226c6269c08b5a78c6b6f1481dfbd。

## 候选适配及其依据

报告名称：AgentSafe paper-components / MAPLE AppWorld adaptation，内部 method 仍为 agentsafe_full。沿用共同 AppWorld action-selection proxy 的 8 agents、3 rounds、200 tasks、15 seeds 42–56、chain/tree/star/random、Qwen/Gemma 主模型和固定 Qwen judge，合计 120 cells / 24,000 次任务评估；这不复现原论文数字，也不是 native AppWorld 执行。

3 条 criterion 分别描述任务相关性、事实/身份有据可依和不把记忆/同伴文本升级为指令。它们受论文对无关信息、错误信息、身份攻击的概念启发，具体措辞和组合完全由本次适配编写。必须保留候选身份，不能称为论文原指令库。原始 cosine 接近程度并不等价于逻辑满足这些要求。

编码器拟用共享 Qwen3-Embedding-8B，逐向量 L2 归一化。保持全部 criterion 严格过阈值，不添加启发式 bypass。按每轮 review_interval=1，当前独立提示语按完整记录反思并移至 holder-local junk；公开 demo 是逐行删除，两者有区别。

已直接调用固定官方 initialize_relations，生成 15 个 seed 的 8-agent policy，每个包含 56 条双向对称关系。路径为 /mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/appworld-candidate-policies/seed42/maple-policy.json 至 seed56，同目录 manifest.json 含哈希。数字 ID、新随机实例、自身权限 4 是宿主约定。同 seed 策略跨模型/拓扑共享。

## 预先固定的校准规则

不填任意默认 theta。候选使用独立官方 CSQA train（/mnt/public/data/wj/baseline-training-inputs/csqa-train/train-00000-of-00001.parquet，及 provenance.json），按 SHA256("agentsafe-candidate-v1:42:" + 官方题目 ID) 排序，前 32 题校准、接下来 32 题验证；题目先验去重和资产 hash 仍需重验。每题固定生成题干+选项、题干+官方正确答案两种干净片段，每个 split 各 64 条干净片段及 64 条对应人为污染片段，只需 embedding。

为直接约束 all-criteria 带来的整体误拒，分数定义为每条消息全部 criterion cosine 的最小值，而不是分别调每个 criterion。N=64、k=floor(0.01N)=0，theta=nextafter(sorted_scores[k], -infinity)。仅允许有限 [-1,1] theta；无合法值则停止。这在当前校准样本上保证 criterion 误拒率不超过 1%，避免在校准数据上几乎全拒；不能保证未见 AppWorld 数据的保留率。独立验证集报告误拒率且不重新调 theta；若明显退化，不进入正式测试，而应另立开发版协议，不能看测试集选阈值。

该规则只校准干净片段保留率，不优化攻击成功率。CSQA 是跨域样本，且裸片段不同于真实记忆和同伴文本，仍有分布偏移；低阈值可能使 cosine 条件几乎全放行。身份、权限、反思条件继续独立生效。正式报告须同时披露 admission criterion 拒绝率和其它拒绝原因，不能把一个近乎无效的 cosine 条件解释成有效检测。

完整候选列明 required artifacts：题目 ID/文本 hash、数据和去重证据、criterion hash、编码器部署身份、所有分数、阈值与冻结时间。校准之后再跑两个主模型各两题的真实组件协议检查，之后才可以执行全量。

## 现有启动参数与最小身份区分

现有 tools/run_appworld_matrix.py 已支持 --methods agentsafe_full、--agentsafe-policy、--agentsafe-criteria、--agentsafe-threshold。当前一个调用只能传一份 policy，故每个 seed 单独调用；不必扩大 CLI 改动。

单模型/seed 的全拓扑参数为 --phase pilot --tasks 200 --profile strict --methods agentsafe_full --seeds 42 --topologies chain tree star random --agentsafe-policy <seed42 policy> --agentsafe-criteria configs/baselines/agentsafe_appworld_candidate.criteria.json --agentsafe-threshold <开发校准值>，再配 bundle/services/backbone/task-service/judge-service/fresh-run-root。省略 --execute 仅准备清单，正式执行另加 --execute 并受共用服务器容量约束。每 cell 独立内存与 sidecar。

身份区分只需在冻结的 matrix/run 元数据写入 profile=paper_components_appworld、上述 reporting_label、完整候选/校准清单的路径和哈希、review_interval=1。现有组件默认已经是 1。不需要改变防御判定逻辑，也不需要用新的数值代替 theta 的开发校准。

本子任务只修改 AgentSafe 专属实现、配置、测试与文档，以及经主代理协调分配的 full_runtime 参数传递；没有修改 INFA、GUARDIAN 或现有实验进程，也没有提交或推送。


## 已完成的开发校准（2026-09-29）

预算修订在任何 embedding 或目标测试结果之前记录。32+32 题，259 个 embedding 文本（256 片段+3 criterion），17 个 embedding 请求，另 1 个 /models 请求，无重试、无 chat 调用。五个 held-out 文件仅验证原始字节 hash，未解析答案或标签。

冻结 theta=0.18154229499203256。calibration 与 validation 各自的 64 条干净片段均为 0 误拒；各自的 64 条人为污染片段均为 0 检出，两种污染类型均为 0/32。余弦准入条件在本次开发样本上全部放行，没有区分能力。这不等于完整方法没有作用：身份、关系权限、分层记忆、逐轮反思仍照常执行；但不能把余弦条件宣传为经验证有效。没有为追求检出率重新改写 criterion 或用目标测试集调 theta。64 条片段来自 32 题，不能把经验零误拒说成总体 FPR≤1%。

冻结目录：
/mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/calibration-csqa32-seed42-20260929/

其中 policy.json、criteria.json、calibration-manifest.json 可以直接用于 seed42，另有题目 ID/文本 hash、开发片段、分数和 API 计数日志。模型 ID 为 Qwen/Qwen3-Embedding-8B；权重 revision 未取得，明确记 null。

新 paper_v2_adapted profile 验证 criteria 文件 SHA、阈值、review_interval、精确 model ID，以及实时 criterion embedding 对冻结向量的 cosine 差≤1e-5；检查通过后使用冻结向量（同一 L2 归一化运算）保证校准与运行时 criterion 几何相同。旧 paper_components 默认保留。full_runtime 已注册 --agentsafe-profile 和 --agentsafe-calibration-manifest 并传递 embed_model；guard.provenance 使用 reporting_label=AgentSafe (full-component adaptation)。

权限代码原第 254–260 行（新增 profile 后行号后移）的条件已经是 assessment.level <= clearance，不需从 v1 改为 v2。具体适配差异是 _clearance 按 owner→recipient 的关系表取值，而非全局接收者等级；shared admit 的 recipient=None 只延后权限检查到实际 read/route，未取消权限控制。

主编排可用 --agentsafe-profile paper_v2_adapted --agentsafe-calibration-manifest <冻结目录>/calibration-manifest.json --agentsafe-policy <冻结目录>/policy.json --agentsafe-criteria <冻结目录>/criteria.json --agentsafe-threshold 0.18154229499203256 --agentsafe-review-interval 1 执行两模型协议检查；通过之后才可统计正式 200 题矩阵。
