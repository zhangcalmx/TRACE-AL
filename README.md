# 结直肠癌根治术吻合口漏风险决策支持（研究原型）

Explainable, rule-based decision-support prototype for anastomotic-leak risk
stratification after curative colorectal cancer surgery.

## 功能

- **风险评估**：录入患者与手术字段，输出规则优先级、触发依据与证据链
- **证据问答**：基于项目证据库的检索问答（无证据不作答）

## 声明

研究原型，未完成临床验证。输出仅供研究参考，不能替代临床医生判断，
不构成诊断或治疗建议。请勿在界面中录入任何可识别患者信息。

## 本地运行

```bash
pip install -e ".[dev,streamlit]"
streamlit run app/streamlit_app.py
```
