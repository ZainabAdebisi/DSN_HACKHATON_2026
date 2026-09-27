# DSN Bootcamp Qualification Hackathon 2026 — ML Track

Solution for the DSN Mart product-store sales prediction challenge, used as the qualifying
hackathon for the DSN AI Bootcamp.

## Problem
Predict `total_sales` for each product-store combination in `test.csv`, using product attributes
(category, price, weight) and store attributes (format, size, age, location tier). Regression
problem, scored by RMSE.

## Approach
1. **Explore & Understand** — inspected shapes, dtypes, and missing values across train/test.
2. **Analyse** — found messy `product_category` text (48 raw values → 16 real categories),
   structural missingness in `store_size` (whole stores, not random rows), and identified
   `store_format` and `product_price` as the strongest sales drivers, with an interaction between
   the two (price's effect on sales depends on store format).
3. **Clean** — normalised category casing, imputed `product_weight_kg` from other rows of the same
   product, encoded missing `store_size` as its own category, flagged likely-artifact zero
   shelf-visibility values.
4. **Engineer** — `price_relative_to_format` (price relative to its store format's average) and
   K-fold-safe target encoding for `product_code` and `store_code`.
5. **Model** — compared Ridge, Random Forest, and Gradient Boosting via 5-fold CV; tuned the tree
   models with `RandomizedSearchCV`; used a 50/50 blend of the tuned models as the final model.
6. **Predict** — generated `submission.csv` from the final blended model.

## Results
| Stage | CV RMSE |
|---|---|
| Raw baseline | ~1130 |
| After cleaning | ~1089 |
| + price × format feature | ~1085 |
| + product target encoding | ~1080.5 |
| + store target encoding | ~1079.4 |
| + hyperparameter tuning (final) | **~1076** |

Public leaderboard score: **1068.39**

## Files
- `DSN_ML_Track_Solution.ipynb` — full notebook: EDA, cleaning, feature engineering, modelling,
  and final write-up.
- `submission.csv` — final predictions.
- `requirements.txt` — Python packages needed to run the notebook.

## How to run
1. Place `train.csv` and `test.csv` in the same folder as the notebook (from the competition's
   Kaggle Data tab — not included here due to competition rules).
2. `pip install -r requirements.txt`
3. Run all cells top to bottom.

## Author
Zainab Salman
