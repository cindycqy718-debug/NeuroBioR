# NeuroBioR 实验复现说明

## 一、复现目标

本发布包用于复现论文中固定 Agent + 不同 LLM 主干的 TEST 集分类结果。被比较的是
不同 LLM 在同一 Agent、同一提示、同一输入输出契约、同一测试样本和同一评分器下
合成信号分类器的能力；Agent 本身不随模型变化。

![完整复现数据流](assets/reproduction_flow.svg)

## 二、数据与任务

| 数据集 | 任务 | TEST 样本数 | 单样本形状 | 类别 |
|---|---|---:|---|---|
| APAVA | HC / AD 来源队列分类 | 1,431 | `[256,16]` | 2 类 |
| TDBRAIN | HC / PD 来源队列分类 | 960 | `[256,33]` | 2 类 |
| ADFTD | HC / FTD / AD 来源队列分类 | 14,648 | `[256,19]` | 3 类 |

这里的统计单位是一个经过 Medformer 流程处理的一秒 EEG 窗口。数据按受试者独立
划分，LLM 方法只在官方 TEST 集上推理，不接触三个划分中的标签。

## 三、准备目录

```text
工作目录/
├── NeuroBioR-Reproducibility/
└── NeuroBioR_Aligned_TEST_Data_v1.1/
```

百度网盘包完整提供 APAVA 和 ADFTD 对齐测试数据。TDBRAIN 受原数据集访问条款约束，
公开包只提供说明；仓库保留其完整冻结预测和统计表，但重新执行信号级推理需要研究者
自行获得授权数据。

## 四、安装

Linux/macOS：

```bash
cd NeuroBioR-Reproducibility
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

Windows PowerShell 激活命令改为：

```powershell
.venv\Scripts\Activate.ps1
```

## 五、一键验证

```bash
python scripts/validate_release.py \
  --data-root ../NeuroBioR_Aligned_TEST_Data_v1.1/data
```

该命令会检查：

1. 数据确实标记为 TEST-only；
2. 公开输入与私有评分标签逐样本同序；
3. 所有 EEG 批次的形状、类型和数值有效；
4. 5 个模型 × 3 个数据集 × 5 次请求，共 75 个运行目录完整；
5. APAVA 和 ADFTD 共 50 个概率矩阵可以重新计算并严格复现已保存的六项指标。

## 六、重新计算结果表

```bash
python scripts/score_archived_runs.py \
  --data-root ../NeuroBioR_Aligned_TEST_Data_v1.1/data \
  --output-root reproduced/scored_runs
```

结果会写入逐次运行表和“均值 ± 样本标准差”汇总表。论文使用的六个指标是 Accuracy、
macro Precision、macro Recall、macro F1、macro AUROC 和 macro AUPRC。

## 七、重放一个真实预测器

```bash
python scripts/replay_frozen_predictors.py \
  --data-root ../NeuroBioR_Aligned_TEST_Data_v1.1/data \
  --dataset APAVA \
  --model-tag qwen3_235b_a22b_thinking_2507_v1 \
  --request-index 41
```

这一步会加载保留下来的 `predictor.py`，对全部 1,431 个 APAVA TEST 窗口重新推理，
再与保存的概率矩阵比较硬标签并重新计算指标。它不重新调用收费 API。

## 八、结果在哪里

- 三张主表：`results/tables/*_submission_table.csv`
- 每次请求的原始指标：`results/tables/agent_llm_per_seed_metrics.csv`
- 五次请求的均值与样本标准差：`results/tables/agent_llm_mean_sample_sd.csv`
- 75 个冻结运行：`results/selected_75_runs/`
- ADFTD 失败模式图表：`analysis/adftd_failure_mode/outputs/`

## 九、复现边界

本仓库可以不调用 LLM API，直接验证已经生成的预测器、概率矩阵和论文表。若要从零
重新合成 75 个预测器，则还需要自行配置兼容的商业模型端点；模型服务版本漂移可能
导致新生成代码不同，因此论文结果以仓库中保留的完整运行工件为准。

