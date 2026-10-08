# ML Assignment 1: Polynomial Regression

**Roll number:** BT2024201

This is my submission for the polynomial regression assignment. I got two datasets:

- **var1**: predict a steam turbine's net power score from 6 control parameters.
- **var2**: predict a subterranean thermal anomaly score from 3 spatial coordinates.

Only polynomial regression was allowed (no random forests, boosting or neural nets), so most of the work was picking the right degree and regularization. I used 5-fold cross-validation to compare everything.

## How the models evolved

| Stage | Script | What I tried | var1 R² | var2 R² |
|:---|:---|:---|:---:|:---:|
| 1. Baseline | `model1_baseline.py` | The hints from the PDF (var1: degree 3 on x1–x3, var2: degree 4 on x1 only) | 0.2050 | 0.0606 |
| 2. OLS sweep | `model2_ols_sweep.py` | Plain OLS with all features (var1: degree 4, var2: degree 8) | 0.9287 | 0.9943 |
| 3. Regularized | `model3_regularized.py` | var1: degree 5 + Lasso, var2: degree 9 + Ridge | 0.9689 | 0.9948 |
| Final | `train_predict.py` | Same as stage 3 with tuned alphas (Lasso 0.0030, Ridge 0.0106) | **0.9689** | **0.9948** |
| 4. Overfit demo | `model4_overfit.py` | Degree 8 / degree 15 OLS with no regularization | 0.5433 | -0.3188 |

CV MSE for the final models is 0.3068 (var1) and 0.2389 (var2).

## Why I ended up with these models

**The PDF hints didn't work for my data.** Using only x1–x3 for var1 and only x1 for var2 threw away real signal. All the features turned out to matter, and using all of them took R² from about 0.2 and 0.06 to above 0.92.

**Lasso for var1.** Degree 5 on 6 features gives 462 terms, and most of those high-order cross terms don't mean much physically. Lasso zeroed out 335 of them and kept 127, and it beat Ridge in CV.

**Ridge for var2.** Degree 9 on 3 features gives 220 terms. The temperature field should vary smoothly in all directions, so I didn't want to knock out individual terms. Ridge shrinks the weights but keeps them all, which gave the best CV score.

**Model 4 was on purpose.** I wanted to see where it breaks. With degree 15 OLS on var2, training R² was about 0.999 but validation R² fell to -0.32 and the weights blew up to around ±4,956. That's classic overfitting, and it's why I stuck with regularized models for the final submission.

## Figures

All plots are in `figures/`:

- `model_comparison.png`: R² and MSE across the stages
- `bias_variance_tradeoff.png`: validation error vs. polynomial degree
- `actual_vs_predicted.png`: out-of-fold predictions vs. true values
- `residual_diagnostics.png`: residual plots and error distributions
- `sparsity_and_coefficients.png`: Lasso sparsity on var1, Ridge vs. OLS weights on var2

## Running the code

Needs `numpy`, `pandas`, `scipy`, `scikit-learn` and `matplotlib`.

```bash
# final predictions
python3 train_predict.py

# diagnostic plots
python3 generate_plots.py

# individual experiment stages
python3 model1_baseline.py
python3 model2_ols_sweep.py
python3 model3_regularized.py
python3 model4_overfit.py

# build the PDF report
python3 build_pdf.py
```

## Files

- `BT2024201_pred_var1.csv`, `BT2024201_pred_var2.csv`: final predictions (1000 rows each, `y` column, no index)
- `BT2024201_Report.pdf` (also `REPORT.pdf`): the submission report
- `BT2024201_train_*.csv`, `BT2024201_test_*.csv`: the provided data
