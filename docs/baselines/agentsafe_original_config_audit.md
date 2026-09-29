# AgentSafe 原配置审计与 AppWorld 候选

核查日期：2026-09-29。完整原论文配置不能从已检查公开资产恢复。用户已接受明示的完整组件适配。独立开发校准已完成，15 个官方生成器关系实例已准备；正式评测由主编排在协议检查后启动。

## 公开来源

已检查 [arXiv v1](https://arxiv.org/html/2503.04392v1)、[v2](https://arxiv.org/html/2503.04392v2)、两版 TeX 附件和 [官方仓库](https://github.com/junyuanM/Agentsafe/tree/cc253ad48532fa6614a27557587086cfb87968ed)。GitHub 只有 main，没有 releases/tags/notebooks；13 次提交的全部文件树和 10 个不同 Python 源码版本均已检查。附件是论文 TeX、样式、引用和图像，没有实验实现或参数文件。

公开主表为 RIOH/WCEI、6 个普通 agent 加 1 个 attacker、10 轮。Eq.8–9 定义“每个 criterion cosine 严格大于 theta”，但未给完整 criterion、theta 或准入编码器。完整身份抽取/匹配提示语、原随机关系实例和完整推理参数也未恢复。

v2 在 Eq.3 和 Algorithm 1 把 v1 的发送者/接收者权限比较改为“消息安全级别不高于接收者权限”。Algorithm 1 的源文件括号把检测置于每轮循环内，支持按 review_interval=1 实施；附录 H 的 5/10/15/20/25/30 是报告交互轮数，不能当检测间隔。表格实际运行的独立数值周期未披露。

process.py 的 text-embedding-ada-002 和 0.2 用于攻击结果评价，不在 agents.py 的准入链路。正文约 0.7 是报告相似度结果，也不是准入 theta。

## 文件与审计证据

- configs/baselines/agentsafe_original_configuration.audit.json：官方可恢复字段、未知字段、SHA-256、历史核查与证据路径。
- configs/baselines/agentsafe_appworld_candidate.json：完整可审阅候选；official_configuration_recovered=false；当前单条请求校准完成后 runnable=true，theta=0.182565380021364；旧批量版本已归档。
- configs/baselines/agentsafe_appworld_candidate.criteria.json：三条独立编写的候选 criterion，非作者发布资产。

原始公开响应和论文附件保存在服务器 /mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/source-audit-20260929/。v1 tar SHA-256 为 5251a33fa61c60b323d8b947fa17a34c7e42a9d377986a09f398fa85eb604517，v2 为 2b6a153856b5a2001adf14fc21f8e292e6f226c6269c08b5a78c6b6f1481dfbd。

## 候选适配及其依据

报告名称：AgentSafe (full-component adaptation)，内部 method 仍为 agentsafe_full。沿用共同 AppWorld action-selection proxy 的 8 agents、3 rounds、200 tasks、15 seeds 42–56、chain/tree/star/random、Qwen/Gemma 主模型和固定 Qwen judge，合计 120 cells / 24,000 次任务评估；这不复现原论文数字，也不是 native AppWorld 执行。

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

身份区分只需在冻结的 matrix/run 元数据写入 profile=paper_v2_adapted、上述 reporting_label、完整候选/校准清单的路径和哈希、review_interval=1。现有组件默认已经是 1。不需要改变防御判定逻辑，也不需要用新的数值代替 theta 的开发校准。

本子任务只修改 AgentSafe 专属实现、配置、测试与文档，以及经主代理协调分配的 full_runtime 参数传递；没有修改 INFA、GUARDIAN 或现有实验进程，也没有提交或推送。


## 初版批量校准（2026-09-29，已由单条协议版本替代）

预算修订在任何 embedding 或目标测试结果之前记录。32+32 题，259 个 embedding 文本（256 片段+3 criterion），17 个 embedding 请求，另 1 个 /models 请求，无重试、无 chat 调用。五个 held-out 文件仅验证原始字节 hash，未解析答案或标签。

冻结 theta=0.18154229499203256。calibration 与 validation 各自的 64 条干净片段均为 0 误拒；各自的 64 条人为污染片段均为 0 检出，两种污染类型均为 0/32。余弦准入条件在本次开发样本上全部放行，没有区分能力。这不等于完整方法没有作用：身份、关系权限、分层记忆、逐轮反思仍照常执行；但不能把余弦条件宣传为经验证有效。没有为追求检出率重新改写 criterion 或用目标测试集调 theta。64 条片段来自 32 题，不能把经验零误拒说成总体 FPR≤1%。

冻结目录：
/mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/calibration-csqa32-seed42-20260929/

其中 policy.json、criteria.json 的原始字节保留；该旧 calibration-manifest.json 已停用于启动，另有题目 ID/文本 hash、开发片段、分数和 API 计数日志。模型 ID 为 Qwen/Qwen3-Embedding-8B；权重 revision 未取得，明确记 null。

新 paper_v2_adapted profile 验证 criteria 文件 SHA、阈值、review_interval、精确 model ID，以及实时 criterion embedding 对冻结向量的 cosine 差≤1e-5；检查通过后使用冻结向量（同一 L2 归一化运算）保证校准与运行时 criterion 几何相同。旧 paper_components 默认保留。full_runtime 已注册 --agentsafe-profile 和 --agentsafe-calibration-manifest 并传递 embed_model；guard.provenance 使用 reporting_label=AgentSafe (full-component adaptation)。

权限代码原第 254–260 行（新增 profile 后行号后移）的条件已经是 assessment.level <= clearance，不需从 v1 改为 v2。具体适配差异是 _clearance 按 owner→recipient 的关系表取值，而非全局接收者等级；shared admit 的 recipient=None 只延后权限检查到实际 read/route，未取消权限控制。

旧版未完成目标题目：真实 smoke 在三次 criterion embedding 后由几何 gate 阻止；目标 task trace 为 0，chat 调用为 0。


## 校准与运行请求协议修复（2026-09-29）

初版工具对 259 个文本使用 input=list、batch_size=16、encoding_format=float；实际 full_runtime._factory 每次发送 input=单字符串，省略 encoding_format。二者没有附加前缀或截断参数，都使用同一模型和 L2 归一化。端点对批内上下文/位置敏感：三条 criterion 单条请求相对旧冻结向量的 cosine error 分别为 0.0004500396503654、0.0001221214070687、0.0000785371697515。仅加 encoding_format=float、或者把单条改为单元素 list 均不能恢复旧几何；batch3 使第二条精确恢复，但其它两条仍不匹配；batch16 仅重复这三条时，同文本在不同位置也产生不同向量。没有把尚未验证的服务 dtype/kernel 解释当作事实。

三轮 scalar 请求三条向量逐元素一致。最小修复只改校准工具：新增 --transport-mode scalar，强制 --batch-size 1，payload 精确为 {model, input: string}，省略 encoding_format；manifest、preregistration、每次 API journal 记录请求协议。full_runtime 请求不改，1e-5 几何 gate 不改，criteria 不改。新增精确 payload、错误批大小提前拒绝及实际 runtime payload 回归。

新目录为 /mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/calibration-csqa32-scalar-seed42-20260929/。259 embedding + 1 /models 全部完成，另用实际 full_runtime._factory 的 3 次 criterion embedding 验证构造成功，无 chat。新旧 seed、question IDs、256 段文本 SHA、criteria SHA、五个 held-out/raw-byte hashes、阈值规则和 policy 字节全部一致。此次重算是修复协议不一致；没有使用目标任务结果选择参数。

新 theta=0.182565380021364；calibration 和 validation 仍各 0/64 干净误拒、0/64 污染检出。仅按原预注册分位数规则计算一次，没有按攻击检测率调优。旧版完整保留。

新 manifest SHA-256：2eea503b14134f2e0adf64d8262dbdb2dc5a48fc534b00a685ada11defe5ffc2。请求协议为 input_shape=string、batch_size=1、encoding_format_policy=omitted。编码器权重 revision 仍未知。诊断原始记录位于 embedding-geometry-diagnosis-20260929/；实际组件验证与输入一致性记录在新目录 verification.json。

两模型 smoke 可各 workers=1 并行，总 children≤2；只有两项均通过 completed/exit0/trace2/API 无错无截断及组件事件验证，才能并行启动两项 200 题 pilot。任一 smoke 失败终止对端并阻止全部 pilot，不自动重试。

## 部署数值检查与独立并发验证

单条请求修复后，两个真实组件同时初始化仍因服务端动态批次变化超出旧 1e-5 门槛。该实时 criterion 请求是部署一致性检查；实际消息评分一直使用 manifest 中的冻结 criterion 向量。因此另立有证据的数值检查规则，不改变防御评分阈值，不用任务成绩调参，也不更改公共请求调度或加入排他锁。

在独立验证之前固定规则：检查容差为全部已测 criterion 向量最大绝对 cosine error M 的两倍，硬上限 0.002；测量不足、非有限值或超上限均停止。43 个测量覆盖 scalar、单元素列表、不同批大小/顺序、重复文本、原批量冻结向量及捕获的并发请求。M=0.0004500396503652748，故容差固定为 0.0009000793007305496。各原始向量文件 SHA 和测量记录保存在 numeric-canary-measurements.json，SHA 为 58fc82af66b41c92465c3f5730aca3dbbfff1c7298228f6906708a2bb49d63ae。最早双并发测试只存了通过/失败，未保存原向量，故另补六请求测量后再冻结；没有把未保存的数值当已知证据。

最终目录为 /mnt/public/data/wj/baseline-reproduction-20260929/agentsafe/calibration-csqa32-scalar-canary-seed42-20260929/，manifest SHA 为 d5e865a4bd6f46f31e32954590e5cbb513932754bf2d94b921d689753086b08c。theta 仍为 0.182565380021364；冻结评分向量、criteria、policy、开发文本、分数、原校准预注册文件与上一 scalar 版本逐字节一致。旧两个校准版本全部保留。

冻结后另发六次 criterion 请求，两个真实 full_runtime._factory 并发初始化均通过，最大实际误差 0.0004500396503651638；两实例复用的评分向量 SHA 相同。此次独立验证没有重新决定 M 或容差。证据在最终目录 parallel-canary-validation.json；这是部署检查通过，不是 AppWorld 任务结果。

仅 paper_v2_adapted 可从经哈希校验的有限测量清单读取该容差。运行时核对 2*M 规则、0.002 硬上限、全部 criterion 测量、参考向量标识，记录每条实时 cosine error 和实际容差；无有效测量时仍用 1e-5。大漂移、无限/未知容差、证据变动均拒绝。回归验证容差变化不会改变冻结评分向量、消息分数或评分 theta。

真实模型权重 revision 仍未知，数值检查不能提供权重级身份保证。共享服务后续出现更大漂移会再次停止；消息 embedding 本身也可受共享批次影响，此限制须保留披露。
