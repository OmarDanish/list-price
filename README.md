# ListPrice – ML-Based Property Valuation

## Overview
ListPrice is a machine learning project built to estimate residential property prices using historical sales data. The goal was to explore how ML models could support pricing decisions while balancing accuracy and explainability.

## Problem
Home pricing is often subjective and varies by market, leading to mispriced listings and slower sales. This project explores whether ML models can provide a reliable pricing baseline.

## Approach
- Dataset: ~20,000 property records
- Models: Linear Regression, Random Forest, Gradient Boosting, XGBoost (ensemble)
- Evaluation: RMSE, MAE, cross-validation
- Tools: Python, scikit-learn, Pandas

## Results
- Achieved ~85% accuracy on validation data
- Ensemble models outperformed single-model baselines
- Identified key drivers such as location, square footage, and recency of sales

## Product Learnings
- Model accuracy alone was insufficient; users wanted explanations for pricing outputs
- Simpler models were often preferred for trust and interpretability
- Edge cases (renovations, outliers) required human review

## Next Steps
- Add explainability (SHAP)
- Market-specific tuning
- Human-in-the-loop pricing adjustments
