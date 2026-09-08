# TRACE-AL — 结直肠癌根治术吻合口漏风险决策支持（研究原型）

Explainable, rule-based decision-support prototype for anastomotic-leak risk
stratification after curative colorectal cancer surgery.

TRACE-AL: a rule-anchored, evidence-traceable clinical decision support system for anastomotic leakage after colorectal cancer surgery
**在线演示**：https://trace-al-cdss.streamlit.app （访问码：TRACE-AL）

## 功能

- **风险评估**：录入患者与手术字段，输出规则优先级、触发依据与证据链
- **证据问答**：基于项目证据库的检索问答（无证据不作答）

## 快照范围

本快照包含确定性评分核心（确定性策略评分：15 组件、满分 40、≥12 主预警）、变量字典、证据库与检索配置，以及 LLM 证据锚点审核（advisory audit）。基线比较评估脚本与五种子评估流程未随本快照归档，需要时可联系通讯作者。

## 数据来源与锁存状态（Provenance）

- 规则策略于 **2026-07-27** 锁存，此后未做任何修改；
- 验证数据库为直接导出（n = 1,646），未做任何修改或处理；策略从未使用任何结局数据进行拟合或调优；
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
