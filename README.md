# California Housing Price Prediction Engine

An end-to-end Machine Learning production pipeline built to predict the median house value of California districts using 1990 census metrics. This repository transitions from exploratory data research to an automated, decoupled, and completely idempotent training and batch-inference architecture.


## Key Engineering Milestones
* **Data Leakage Mitigation:** Built a robust data topology ensuring zero leakage across sampling, imputation, and feature scaling phases.
* **Stratified Split Parity:** Resolved standard random sampling bias by implementing a stratified distribution lock based on engineered median income categories.
* **Production-Grade Pipelines:** Automated complex numerical and categorical vector transformations natively via Scikit-Learn `ColumnTransformer` blocks.
* **Idempotent Inference System:** Engineered a state-persistent execution flow that serializes model and pipeline weights, enabling isolated batch inference scripts without mutating original feature data schemas.


## Pipeline Architecture & Data Flow

The project layout separates transformation pipelines to ensure categorical variables are dynamically vectorized in parallel to continuous numeric sequences, preventing target or spatial dimension bias.

```text
                           [ housing.csv ]
                                  │
                     Does model.pkl exist? (os.path)
                    ├── NO ─────────────────────── YES ──┐
                    ▼                                    ▼
             [ TRAINING PHASE ]                  [ INFERENCE PHASE ]
  ├── Stratified Data Partitioning        ├── Deserialize Model & Pipeline
  ├── Isolate Target Column (y)           ├── Read Unlabeled input.csv
  ├── Fit & Save Preprocessing Pipeline   ├── Execute pipeline.transform()
  ├── Fit & Save RandomForest Model       ├── Generate Predictions Vector
  └── Output: .pkl files + input.csv      └── Output: Export clean output.csv
```

## Data Engineering & Preprocessing

### 1. Stratified Partitioning (Handling Sampling Bias)
A simple random shuffle risks under-representing critical income segments, which serves as the most heavily weighted metric driving property values. To enforce historical and statistical parity:
* A helper feature (`income_cat`) is engineered by discretizing continuous income bounds: `[0.0, 1.5, 3.0, 4.5, 6.0, np.inf]`.
* `StratifiedShuffleSplit` divides the data into an 80/20 train/test split, ensuring both segments mirror the identical target distribution density of the true population.
* The helper category is subsequently dropped to retain clean data schemas.

### 2. Parallel Stream Preprocessing
Transformations are organized through a central `ColumnTransformer` to enforce reproducible transformations on both continuous and discrete feature paths:
* **Numerical Stream:** Targets missing values via a median `SimpleImputer` (resolving the 1% missingness found in `total_bedrooms`) and applies unit-variance normalization via `StandardScaler`.
* **Categorical Stream:** Converts nominal strings (`ocean_proximity`) into binary indicator arrays using `OneHotEncoder(handle_unknown="ignore")`. This explicitly prevents models from learning arbitrary sequences inherent to basic ordinal encoding.

---

## Model Performance & Overfitting Diagnostics

Estimators were systematically evaluated via a **10-Fold Cross-Validation** sequence to extract reliable out-of-fold metrics regarding generalization performance and variance:

1. **Linear Regression (Baseline):** Suffered from high bias due to an inability to interpret the non-linear relationship between spatial coordinates and pricing cliffs.
2. **Decision Tree Regressor:** Overfitted significantly on raw instances, returning an artificial **$0.00 training RMSE** while displaying high variance and poor scaling during cross-validation rounds.
3. **Random Forest Regressor (Champion):** Built an ensemble of 100 decorrelated decision estimators, reducing generalization variance by averaging local patterns across out-of-fold loops.

---

## Repository Structure

```plaintext
├── data/
│   └── housing.csv               # Raw source dataset (Git ignored)
├── notebooks/
│   ├── California_house_prices_Analysis.ipynb
│   ├── Stratified_shuffle_split.ipynb
│   ├── Visulizing_data.ipynb
│   ├── Handling_missing_data.ipynb
│   ├── Feature_scaling.ipynb
│   ├── Pipeline_construction.ipynb
│   └── Consolidating_pipeline.ipynb
├── model.pkl                     # Serialized RandomForest weights
├── pipeline.pkl                  # Serialized ColumnTransformer states
├── input.csv                     # Unlabeled test partition features for batch runs
├── output.csv                    # Inference results appending predictions
├── main.py                       # Idempotent production execution script
└── .gitignore                    # Ensures binaries/data files are untracked
