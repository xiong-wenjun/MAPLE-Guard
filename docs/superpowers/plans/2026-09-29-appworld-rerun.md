# AppWorld 首轮复现与严格对照实现计划

**目标：** 使用用户固定的五份 200 条数据，先验证 MAPLE 现有 AppWorld 行动选择协议，再按冻结的方法矩阵启动真实模型实验。
**架构：** 原论文代码配置与修订公平对照分开保存。所有代码、数据、模型和运行记录留在服务器 /mnt/public/data/wj；本地只保存审计报告。
**技术栈：** Python unittest、现有 MAPLE stream runner、SSH、OpenAI-compatible Qwen/Gemma/embedding 服务。

- [ ] 添加严格的数据封装验证与 AppWorld --benchmark-bundle 入口。拒绝重复 ID、数量错误和 benchmark 不匹配；保留原始顺序与原生 task ID，标注 action-selection proxy，绝不伪装原生工具评测。
- [ ] 新增 tests/test_benchmark_bundle.py：数量/重复校验、AppWorld 不访问默认全数据索引、CSQA 问题与选项转换、任务顺序/哈希、proxy summary 范围。
- [ ] 修复服务认证：chat 与 embedding 使用各自环境变量，密钥不进入 CLI、Git 或运行 manifest；添加 mock HTTP 回归测试。
- [ ] 为 Challenger/G-Safeguard/GUARDIAN 增加严格运行路由；模型或 judge 故障必须失败，禁止 allow-all 伪结果。实现与测试由独立子任务负责。
- [ ] 保存 14 个方法配置：主表 11 项及机制对照 3 项。保留原主表；所有 Gemma arms 固定同一 Qwen judge。AgentSafe、INFA、GUARDIAN 不满足资源与方法完整性时明确阻塞。
- [ ] 服务器探测与 API smoke：确认实际 model ID、embedding 维度、token 输出与无截断。禁止擅自停止现有模型服务或将 Qwen 标成 Gemma。
- [ ] 先运行 MAPLE 与 No Defense 的论文配置小规模检查，验证 trace、memory 与 summary 后再启动可运行方法的 200 条任务。
- [ ] 运行 unittest 全套与 CLI manifest 验证；保存 commit、数据 SHA256、参数、运行状态和失败原因。实验结果不由单元测试替代。

验证命令：PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 INFA_NATIVE_SOURCE=/mnt/public/data/wj/baseline-references/INFA-Guard /mnt/public/data/wj/venvs/maple-baselines/bin/python -B -m unittest discover -s tests
