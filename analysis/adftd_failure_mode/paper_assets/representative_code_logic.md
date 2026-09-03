# Representative generated-code logic

## GLM-5, seed 41

- Original file:
  `results/selected_75_runs/glm5_reasoning_v1/ADFTD/seed41/predictor.py`
- Observed output on frozen TEST:
  HC `0.0137%`, FTD `0%`, AD `99.9863%`.

The following is the exact score-mapping block from the saved predictor:

```python
# Healthy control: preserved alpha, high complexity, good connectivity
healthy = (
    m_alpha * 4.0 +
    m_entropy * 3.0 +
    m_corr * 2.0 +
    (1.0 - slow_power) * 2.0 +
    max(0.0, alpha_peak - 9.0) * 0.2
)

# FTD: focal frontal-temporal changes, spatial heterogeneity
ftd = (
    alpha_var * 5.0 +
    delta_var * 3.0 +
    m_delta * 2.5 +
    m_theta * 1.5 +
    (1.0 - m_corr) * 0.8
)

# AD: diffuse slowing, reduced alpha, reduced connectivity
ad = (
    slow_power * 3.5 +
    (1.0 - m_alpha) * 3.0 +
    (1.0 - m_entropy) * 2.5 +
    (1.0 - m_corr) * 1.5 +
    theta_alpha_ratio * 1.5 +
    max(0.0, 10.0 - alpha_peak) * 0.25
)

scores = np.array([healthy, ftd, ad], dtype=np.float64)
exp_scores = np.exp(scores - np.max(scores))
probs = exp_scores / np.sum(exp_scores)
```

This excerpt demonstrates the paper's program-level claim without implying an
internal LLM mechanism: several independently constructed “abnormal” terms are
added to the AD score, whereas the FTD score depends strongly on spatial
heterogeneity. On the actual independently normalized one-second TEST windows,
this uncalibrated score geometry makes the FTD region empirically unreachable
and the AD region dominant.

The other four complete predictors remain available under
`results/selected_75_runs/glm5_reasoning_v1/ADFTD/seed42` through `seed45`;
their summarized decision consequences are reported in the analysis tables.

