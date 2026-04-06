# Using Machine Learning to Assess Going Concern Assumptions

A machine learning project that predicts company bankruptcy from financial statement ratios, developed as coursework for CSM010 (Applied Machine Learning).

## Overview

Auditors are required under ISA 570 to evaluate whether a company can continue as a going concern. This process involves assessing multiple financial ratios collectively — a task that is complex, subjective, and historically unreliable. Research suggests auditors issue clean opinions to approximately 42% of companies that file for bankruptcy within the following year.

This project explores whether machine learning classifiers can outperform that benchmark by learning patterns from historical financial data.

## Dataset

**Taiwanese Bankruptcy Prediction Dataset** — UCI Machine Learning Repository

- Collected by the Taiwan Economic Journal (TEJ) covering company data from 1999–2009
- 6,819 company instances, 96 financial attributes (profitability, liquidity, leverage ratios, etc.)
- Heavily imbalanced: only 3.2% of companies are bankrupt (222 bankrupt vs. 6,597 non-bankrupt)
- No missing values; all features are numeric

## Approach

### Preprocessing
- **Feature scaling:** `StandardScaler` applied for models sensitive to feature magnitude; tree-based models trained on unscaled data
- **Feature selection:** Two-stage process — automated selection using Random Forest feature importances (top 15 features), followed by manual elimination of redundant/inverse features using domain knowledge
- **Class imbalance:** Up-sampling (×2) preferred over down-sampling to avoid discarding information

### Models Evaluated
- Logistic Regression -> Baseline linear model
- Support Vector Machine -> Scaled features 
- Decision Tree -> Best overall performer 
- Random Forest -> Best precision 
- Linear Discriminant Analysis -> Scaled features 
- Gaussian Naive Bayes -> Strong initial recall 

**Primary metric: recall** — in an audit context, false negatives (missed bankruptcies) are more costly than false positives.

## Results

The **Decision Tree Classifier** achieved **81% recall** on the test set, substantially better than the ~58% implied by historical audit industry data. Random Forest delivered the best precision; Decision Tree won on recall and AUC.

The decision tree's rules were extracted and visualised (see Appendices in the notebook) to make predictions interpretable to non-technical auditors.

## Deployment Considerations

This model is scoped to the **audit planning phase** (ISA 570.10 — identifying conditions that may cast doubt on going concern). It is not intended to replace the full audit assessment, but to flag companies warranting deeper investigation.

Key challenges for real-world deployment:
- **Regulatory:** Audit documentation standards don't currently accommodate ML model outputs
- **Interpretability:** Most auditors lack ML background; model transparency is essential
- **Data:** The dataset covers Taiwanese companies only; broader applicability would require more diverse data sources (e.g., Thomson Reuters, Moody's)

## Tech Stack

- Python 3.10
- `scikit-learn` — classification, preprocessing, feature selection, resampling, metrics
- `pandas` — data manipulation
- `matplotlib` — feature importance plots, decision tree visualisation

---
💻 [View the notebook](https://github.com/thabanibiyela/aml-going--concern/blob/master/csm010-aml-coursework-TB203.ipynb)
---
## Files

```
.
├── [csm010-aml-coursework-TB203.ipynb](csm010-aml-coursework-TB203.ipynb)   # Main notebook
└── datasets/
    └── data.csv                         # Taiwanese bankruptcy dataset
```
