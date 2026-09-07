# TRACE-AL — 结直肠癌根治术吻合口漏风险决策支持（研究原型）

Explainable, rule-based decision-support prototype for anastomotic-leak risk
stratification after curative colorectal cancer surgery.

**论文**：TRACE-AL: a rule-anchored, evidence-traceable clinical decision support system for anastomotic leakage after colorectal cancer surgery（投稿中）
**在线演示**：https://trace-al-cdss.streamlit.app （访问码：TRACE-AL）

## 功能

- **风险评估**：录入患者与手术字段，输出规则优先级、触发依据与证据链
- **证据问答**：基于项目证据库的检索问答（无证据不作答）

## 与论文的对应关系

- **确定性评分策略**：`configs/rule_policy_v1_4.yaml` —— 15 个计分组件（10 条基础规则 + 1 项营养风险复合 + 4 项交互加分），理论满分 40 分，主要预警阈值 ≥12，自 2026-07-27 冻结后未做任何修改
- **变量字典**：`configs/clinical_variables.yaml` —— 其中前 15 个患者字段与前 16 个手术/吻合字段即论文的 **31 个核心输入变量**（顺序与论文 Supplementary Table S2 一致）；其余字段为研究扩展采集字段，不参与评分
- **规则触发条件与逐条证据引用**：`configs/rules.yaml`
- **证据库**：275 个经核验证据片段（证据等级 A/B/C/D/E 分布 30/84/155/1/5），见 `data/seed/evidence_chunks.jsonl` 与 `data/processed/lightrag_index_*/`（知识图谱 1,307 实体 / 1,726 关系）
- **检索配置**：mix 模式；实体/关系 top-k = 40；文本片段 top-k = 20；相似度阈值 0.20；未启用重排序
- **LLM 证据锚点审核（advisory audit）**：`src/anastomotic_leak_risks_agent/safety_agent.py` —— 审核分值 0–1 与 accept/disagree 结论仅供建议，不改变规则分级；不一致结论随规则分级保留供复核

## 快照范围

本快照包含确定性评分核心（冻结 v1.4 策略评分：15 组件、满分 40、≥12 主预警）、变量字典、证据库与检索配置，以及论文所述 LLM 证据锚点审核（advisory audit；2026-09 依据研究者记录重建）。基线比较评估脚本与五种子评估流程未随本快照归档，需要时可联系通讯作者。

## 版本沿革（Provenance）

- 规则策略于 **2026-07-27** 冻结（v1.4 替换 v1.1；布尔触发规则不变），冻结后未做任何修改；
- 验证数据库在质控期随缺失数据修复分次重新导出（内部版本 v1–v7），最终锁定版 v7（n = 1,646）用于论文全部报告分析；策略从未使用任何结局数据进行拟合或调优；
- 原始工程文件遗失后，本仓库快照于 2026-09 依据研究者记录与证据库导出重建；此前自动生成的副本曾将冻结日期误写为 2026-08-27，现依据研究者笔记更正为 **2026-07-27**。

## 声明

研究原型，未完成临床验证。输出仅供研究参考，不能替代临床医生判断，
不构成诊断或治疗建议。请勿在界面中录入任何可识别患者信息。

## 本地运行

```bash
pip install -e ".[dev,streamlit]"
streamlit run app/streamlit_app.py
```

## License

MIT — 见 [LICENSE](LICENSE)
