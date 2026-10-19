# F1 Podium Prediction — MS1

Predict whether a Formula 1 driver finishes on the podium (top 3), using only information known **after qualifying
and before the race starts**.

- **Notebook:** `F1_Podium_MS1.ipynb` (also on Kaggle: https://www.kaggle.com/code/amreladawy2/f1-podium-prediction-ms1)
- **Report:** `MS1_Report.pdf`
- **Data:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by rohanrao, 14 CSV tables

## What the notebook does
1. **EDA before cleaning** — missing values (`msno.matrix`), qualifying coverage per season, how each entry ended.
2. **Cleaning (10 rules)** — `\N` → NaN, key checks, Indy 500 removed, non-starters removed, one row per
   (raceId, driverId), grid 0 → pit-lane start, target `podium`, standings shifted to the previous round,
   nationality → country, status groups.
3. **EDA after cleaning** and the three data questions:
   - **Q1** front row → podium by circuit, 1994–2013 vs 2014–2024
   - **Q2** home advantage, controlling for starting position
   - **Q3** mechanical retirements by constructor, 2014–21 vs 2022–24
4. **Features (12, all pre-race)** — grid, qualifying gap, recent driver/team form (`shift(1)` before rolling),
   standings before the race, home race; leakage check.
5. **Split by season** — train 2003–2019, validation 2020–21, test 2022–24; median fill + `StandardScaler` fit on train.
6. **Models** — Logistic Regression (baseline), Random Forest (default and tuned), Keras neural networks
   (Lab 3 64→32→16 and shallow 64→32); PR-AUC, ROC-AUC, F1, precision, recall, accuracy; threshold chosen on validation.
7. **Ablation**, **preprocessing-order experiments**, **XAI** (permutation importance, SHAP, LIME),
   **inference function** `predict_podium(record)`, conclusions and limitations.
8. **Bonus (section 21)** — the best of 13 tested cleaning/split variants (V13): missing values filled with their F1
   meaning, the two redundant constructor features dropped, validation 2019–21. All 13 runs:
   https://www.kaggle.com/code/amreladawy2/f1-ms1-bonus-cleaning-and-split

## Main results
- Chosen model: Random Forest (tuned), validation PR-AUC 0.719, test PR-AUC 0.684 (grid ≤ 3 rule: 0.474).
- Bonus V13: RF tuned test PR-AUC 0.692 and F1 0.686 (vs 0.684 / 0.642), chosen on validation 2019–21.
- Grid position and qualifying gap carry most of the signal; there is no home advantage.

## How to run
On Kaggle: open the notebook, attach the dataset above and choose **Run All**.
Locally: `pip install -r requirements.txt`, download the dataset and put the 14 CSV files in a `data/` folder next to
the notebook, then run all cells (section 1 looks in `/kaggle/input` first, then in `data/`).
