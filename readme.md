# Machine Learning Project: Standard of Living Analysis

This repository contains a machine learning project analyzing standard of living using the European Social Survey (ESS) dataset. The project uses logistic regression to predict life satisfaction (`stflife`) and evaluates how people feel about household income (`hinctnta`) and overall life satisfaction. It includes data cleaning, feature engineering, model training, and evaluation, achieving an accuracy of 85.1%. Built with Python using pandas, statsmodels, scikit-learn, and visualization libraries.

## Project Overview
- **Dataset**: ESS11-subset.csv (1771 rows, 42 columns).
- **Objective**: 
  - Q1: Assess comfort/difficulty with current household income.
  - Q2: Evaluate life satisfaction considering other factors.
- **Model**: Logistic Regression.
- **Accuracy**: 85.1% on the test set.
- **Tools**: Python, pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels.

## Step-by-Step Process

### 1. **Data Loading and Initial Exploration**
- Load the ESS dataset (`ESS11-subset.csv`) and display its shape (1771 rows, 42 columns).
- Explore variables like `stflife` (0-10 satisfaction scale), `hinctnta` (income satisfaction), `eduyrs`, `agea`, `vote`, and `gndr`.

### 2. **Data Cleaning**
- Check for missing values using `df.isna().sum()` (no missing values found).
- Verify data completeness with `df.count()` (all 1771 rows present).
- Identify and handle non-valid values (e.g., 77, 88, 99 for `eduyrs`, 666+ for `wkhtot`) if needed (assumed handled earlier).

### 3. **Feature Engineering**
- Select relevant features: `eduyrs`, `wkhtot`, `hhmmb`, `agea`, and nominal variables (`vote`, `rlgblg`, `emplrel`, `gndr`).
- Create dummy variables for categorical features using `pd.get_dummies()` with `drop_first=True`.
- Combine numeric and dummy variables into feature matrix `X` and add a constant with `sm.add_constant(X)`.
- Binarize `stflife` (e.g., 0-5 = 0, 6-10 = 1) as the target variable `y`.

### 4. **Data Splitting**
- Split data into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42`:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### 5. **Model Training**
- Train a logistic regression model using `statsmodels` (`sm.Logit`):
  ```python
  logit_model = sm.Logit(y_train, X_train)
  result = logit_model.fit()
  print(result.summary())
  ```
- Alternatively, use `scikit-learn`’s `LogisticRegression` for predictions.

### 6. **Model Evaluation**
- Compute manual metrics:
  - Accuracy: 85.1%
  - Specificity: 5.7%
  - Sensitivity: 99.0%

### 7. **Visualization**
- Visualize the confusion matrix to assess true positives, false positives, etc.
- Analyze model performance based on the heatmap and metrics.

## Results and Interpretation
- **Accuracy**: 85.1%, indicating strong predictive power.
- **Specificity**: 5.7%, showing low ability to correctly identify "Not Satisfied" cases.
- **Sensitivity**: 99.0%, showing excellent ability to identify "Satisfied" cases.
- The model performs well overall, though specificity suggests potential imbalance or threshold adjustment needs.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook ML_Project_Standard_of_living.ipynb
   ```
- Ensure `ESS11-subset.csv` is in the project directory or update the file path.

## Notes
- The dataset is assumed to be pre-cleaned for non-valid values; adjust cleaning steps if needed.
- Use `random_state=42` for reproducibility.
- Consider cross-validation or adjusting the threshold (e.g., 0.5) for better specificity.