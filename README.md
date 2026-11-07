# Boston Housing Price Prediction | Linear Regression

**Codveda Technologies — Machine Learning Internship**  
**Level 1 · Task 2: Linear Regression**

## Overview

This project investigates how housing characteristics relate to median home values using the historical Boston Housing dataset. It implements **Simple Linear Regression** and **Multiple Linear Regression** to meet the core task requirements, interprets their coefficients, and evaluates their predictive performance. A **Polynomial Features + Ridge Regression** pipeline is included as an optional extension to explore nonlinear relationships.

## Objectives

- Load and inspect the dataset, including missing values, duplicates, and descriptive statistics.
- Explore relationships between predictors and median home values using training data.
- Train and evaluate simple and multiple linear regression models.
- Interpret coefficients and investigate multicollinearity.
- Compare models using 5-fold cross-validation, mean squared error (MSE), root mean squared error (RMSE), and R².
- Export and verify the selected extension model and comparison results.

## Dataset

- **Source:** Historical Boston Housing data supplied for the Codveda internship task.
- **Records:** 506; **columns:** 14, including the target.
- **Target:** `MEDV`, median owner-occupied home value, expressed in thousands of historical US dollars.
- **Model inputs:** 12 numerical features. The race-derived `B` variable is excluded from modeling.
- **Split:** 404 training observations (80%) and 102 test observations (20%), using `random_state=42`.

The supplied CSV is **whitespace-delimited and has no header row**. The notebook provides the 14 column names explicitly and locates the dataset in Kaggle's input directory. For a different environment, set `DATA_PATH` to the local CSV path.

## Approach

1. **Data preparation and EDA:** Verify the dataset and inspect relationships with `MEDV` on training data only. In particular, `RM` has a positive correlation (`r = 0.7107`) and `LSTAT` a negative correlation (`r = -0.7364`) with the target.
2. **Simple Linear Regression:** Use `RM` as the sole predictor and interpret the fitted equation.
3. **Multiple Linear Regression:** Train with all 12 selected predictors, examine coefficients, and assess multicollinearity. `RAD` and `TAX` are highly correlated (`r = 0.9125`), with VIFs of 7.62 and 8.93, respectively.
4. **Advanced extension:** Apply second-degree polynomial features to `RM` and `LSTAT`, scale features inside a scikit-learn pipeline, and fit `Ridge(alpha=1.0)`. The regularization strength was fixed, not tuned.
5. **Evaluation:** Compare the approaches using shuffled 5-fold cross-validation on training data, then report results for the held-out test split.
6. **Reproducibility:** Save the fitted extension pipeline and cross-validation comparison, reload them, and verify consistent predictions and data.

## Results

### Cross-validation (training data)

| Model | Mean CV MSE ↓ | Mean CV R² ↑ | CV R² Std ↓ |
|---|---:|---:|---:|
| Simple Linear Regression | 43.2784 | 0.4792 | 0.1484 |
| Multiple Linear Regression | 25.3022 | 0.7069 | 0.0813 |
| Polynomial + Ridge (extension) | **19.4458** | **0.7731** | **0.0744** |

### Held-out test split (102 observations)

| Model | Test MSE ↓ | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|
| Simple Linear Regression | 46.1448 | 6.7930 | 0.3708 |
| Multiple Linear Regression | 22.7777 | 4.7726 | 0.6894 |
| Polynomial + Ridge (extension) | **13.4531** | **3.6678** | **0.8166** |

**Main task outcome:** Multiple Linear Regression improved on the single-feature baseline. The coefficient analysis provides interpretability, but correlated predictors and different feature units limit comparisons of coefficient magnitudes.

**Extension outcome:** Polynomial + Ridge achieved the strongest mean cross-validation performance among the three models. Its test R² of 0.8166 was higher than its mean CV R² of 0.7731; this single 102-sample test split should not be interpreted as a guarantee of future performance.

## Repository Structure

```text
Codveda-House-Price-Prediction/
├── boston-housing-price-prediction.ipynb
├── README.md
├── requirements.txt
└── outputs/
    ├── model_comparison.csv
    └── polynomial_ridge_model.joblib
```

The `outputs/` files are generated under `housing_regression/` in Kaggle's working directory. They can be downloaded and placed in `outputs/` for the GitHub repository. The original dataset is not redistributed in this repository.

## Running the Notebook

1. Obtain the internship's Boston Housing CSV and attach it as a Kaggle dataset, or set `DATA_PATH` in the loading cell to its local path.
2. Install the dependencies listed in `requirements.txt`, if running locally:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `boston-housing-price-prediction.ipynb` in Kaggle or Jupyter.
4. Restart the kernel/session and run all cells in order.
5. Retrieve the exported model and comparison CSV from `housing_regression/`.

**Main libraries:** pandas, Matplotlib, Seaborn, scikit-learn, and joblib.

## Interpretation and Limitations

- The Boston Housing dataset is historical, not representative of present-day housing markets, and is used **for educational purposes only**.
- The race-derived `B` feature was excluded, but other socioeconomic variables may still raise fairness concerns.
- `MEDV` is capped at 50 (in $1,000 units), which limits representation of higher-value homes.
- Linear regression coefficients reflect **conditional associations, not causal effects**. Feature units differ, and multicollinearity can destabilize individual estimates.
- The Ridge extension uses a fixed `alpha=1.0`; this project does not claim that value is optimal.
- The initial test set was examined before the polynomial extension was developed. Although the extension was selected using training-set cross-validation, its subsequent test comparison is **not a fully independent assessment of the overall development process**.
- Performance is based on one train/test split and should be validated more broadly before generalization.

## Conclusion

The project fulfills the core Codveda Level 1 Linear Regression requirements with interpretable simple and multiple regression models, coefficient analysis, and MSE/R² evaluation. The optional polynomial and Ridge extension demonstrates the potential benefit of nonlinear feature engineering while retaining transparent discussion of model limitations and evaluation methodology.
