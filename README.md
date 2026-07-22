## Project Overview: Election Dataset Portugal - Regression

This notebook details a machine learning project focused on predicting `FinalMandates` using the Real-Time Election Results Portugal 2019 dataset. The process involves comprehensive data preprocessing, feature engineering, and the implementation and evaluation of various regression models, particularly focusing on the K-Nearest Neighbors (KNN) Regressor, Linear Regression, and Decision Tree Regressor.

### Objectives

- Understand the dataset structure.
- Identify and handle missing or inconsistent values.
- Detect and address duplicate records.
- Analyze the distribution of numerical and categorical variables.
- Identify and manage outliers.
- Prepare the dataset for statistical analysis and machine learning.
- Build and evaluate regression models to predict `FinalMandates`.

### Preprocessing Steps Performed

#### 1. Data Loading & Initial Exploration
- Dataset loaded from : https://archive.ics.uci.edu/dataset/513/real+time+election+results+portugal+2019 .
- Initial checks performed using `head()`, `shape`, `info()`, and `describe()` to understand data types, dimensions, and basic statistics.

#### 2. Missing Value & Duplicate Handling
- **Missing Values**: Checked using `isnull().sum()`. *No missing values were found*.
- **Duplicate Records**: Checked using `duplicated().sum()`. *No duplicate records were found*.

#### 3. Outlier Detection & Handling
- Outliers were identified using **Boxplots** for numerical features.
- **IQR method** was applied to quantify outliers for each numerical column.
- *Decision*: Outliers were identified but **not removed** or clipped, as they were deemed potentially representative of real-world variations in election data (e.g., large vote counts in major territories/parties).

#### 4. Data Visualization (EDA)
- **Countplot**: Visualized the distribution of political parties (`Party`).
- **Histograms**: Displayed distributions of numerical features to understand their spread and skewness.
- **Boxplots**: Used to visualize the distribution and identify outliers in numerical features.
- **Scatterplots**: Plotted relationships between numerical features and the target variable (`FinalMandates`) to understand correlations.

#### 5. Feature Engineering
- **Correlation Analysis**: Calculated Pearson correlation matrix for numerical features.
- **Correlation Heatmap**: Visualized correlations to identify highly correlated features.
- **Mutual Information**: Calculated Mutual Information (MI) scores between features and `FinalMandates` to determine feature importance.
- **Feature Removal**: 
    - Highly correlated features (`Hondt`, `Mandates`) were removed to reduce multicollinearity.
    - Low mutual information and low correlation features (`TimeElapsed`, `blankVotesPercentage`, `pre.blankVotesPercentage`, `nullVotesPercentage`, `pre.nullVotesPercentage`) were removed.
    - The `time` column was dropped due to high cardinality and its limited relevance to election outcomes.

#### 6. Feature Scaling
- **Standard Scaling**: `StandardScaler` was applied to all remaining numerical features (both `int64` and `float64` types) to standardize their ranges (mean=0, standard deviation=1).

#### 7. Encoding Categorical Columns
- **One-Hot Encoding**: Categorical features (`territoryName`, `Party`) were converted into numerical format using `OneHotEncoder(sparse_output=False)`.

# Model Building

#### 1. Train-Test Split
- The preprocessed data `X` (features) and `y` (target: `FinalMandates`) were split into training and testing sets with a `test_size` of 0.2 and `random_state=33`.

#### 2. K-Nearest Neighbors (KNN) Regressor
- **Baseline KNN Model**:
    - MSE: 0.1363
    - MAE: 0.0236
    - R² Score: 0.9974
- **Boosting with KNN**:
    - MSE: 0.00003
    - MAE: 0.0001
    - R² Score: 0.999999
- **Cross-Validation (K-Fold)**:
    - Cross validation scores: [99.75%, 99.96%, 99.88%, 99.74%]
    - Mean of CV scores: 99.83%
- **Hyperparameter Tuning (Grid Search CV)**:
    - Best Parameters: `{'n_neighbors': 3, 'weights': 'distance'}`
    - Best CV Score: 0.9983
    - MAE: 0.0023
    - MSE: 0.0160
    - R² Score: 0.9997
- **Hyperparameter Tuning (Randomized Search CV)**:
    - Best Parameters: `{'weights': 'distance', 'p': 1, 'n_neighbors': 3}`
    - Best CV Score: 0.9983
    - MAE: 0.0088
    - MSE: 0.0261
    - R² Score: 0.9995

#### 3. Linear Regression Model
- **Baseline Linear Regression Model**:
    - MSE: 13.7068
    - MAE: 0.8314
    - R² Score: 0.7354
- **Boosting with Linear Regression**:
    - MSE: 13.6542
    - MAE: 0.8408
    - R² Score: 0.7365
- **Cross-Validation (K-Fold)**:
    - Cross validation scores: [75.24%, 82.39%, 86.35%, 83.63%]
    - Mean of CV scores: 81.90%
- **Hyperparameter Tuning (Grid Search CV)**:
    - Best Parameters: `{'fit_intercept': True, 'positive': False}`
    - Best CV Score: 0.8428
    - MAE: 0.8314
    - MSE: 13.7068
    - R² Score: 0.7354
- **Hyperparameter Tuning (Randomized Search CV)**:
    - Best Parameters: `{'positive': False, 'fit_intercept': True}`
    - Best CV Score: 0.8413
    - MAE: 0.8314
    - MSE: 13.7068
    - R² Score: 0.7354

#### 4. Decision Tree Regressor
- **Baseline Decision Tree Model**:
    - MSE: 0.0018
    - MAE: 0.0018
    - R² Score: 0.99996
- **Bagging (Random Forest) Model**:
    - MSE: 0.0758
    - MAE: 0.0137
    - R² Score: 0.9985
- **Cross-Validation (K-Fold)**:
    - Cross validation scores: [95.66%, 99.86%, 95.69%, 99.99%, 99.97%]
    - Mean of CV scores: 98.23%
- **Hyperparameter Tuning (Grid Search CV)**:
    - Best Parameters: `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}`
    - Best CV Score: 0.9943
    - MAE: 0.0214
    - MSE: 0.0162
    - R² Score: 0.9997
- **Hyperparameter Tuning (Randomized Search CV)**:
    - Best Parameters: `{'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': None}`
    - Best CV Score: 0.9821
    - MAE: 0.0296
    - MSE: 0.3680
    - R² Score: 0.9929

#### 5. SVM Model
- **Baseline SVM Model**:
    - MSE: 0.0043
    - MAE: 0.0551
    - R² Score: 0.9999
- **Bagging with SVM**:
    - MSE: 0.0069
    - MAE: 0.0575
    - R² Score: 0.9999
- **Cross-Validation (K-Fold)**:
    - Cross validation scores: [99.99%, 99.99%, 99.99%]
    - Mean of CV Score: 99.99%
- **Hyperparameter Tuning (Grid Search CV)**:
    - Best Parameters: `{'C': 100, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'}`
    - Best R² score (CV): 0.9999
- **Hyperparameter Tuning (Randomized Search CV)**:
    - Best Parameters: `{'kernel': 'rbf', 'gamma': 'scale', 'epsilon': 0.01, 'C': 100}`
    - Best R² score (CV): 0.9999

## Deep Learning Models

#### 6. TensorFlow Model
- **ANN Model**:
    - MSE: 0.0111
    - MAE: 0.0245
    - R² Score: 0.9998

#### 7. PyTorch Model (Regression)
- **ANN Model**:
    - The PyTorch model has been adapted for regression. Its architecture now outputs a single continuous value without a final activation function like sigmoid, and it uses Mean Squared Error (MSE) as its loss function.
    - **To see the metrics for the PyTorch model, please run the corresponding cells in the notebook.**

## Evaluation Comparison Table

| Model                                         | MSE      | MAE      | R² Score |
|:----------------------------------------------|:---------|:---------|:---------|
| KNN Regressor (Baseline)                      | 0.1363   | 0.0236   | 0.9974   |
| KNN Regressor (Boosting)                      | 0.0002   | 0.0009   | 0.999996 |
| KNN Regressor (Grid Search CV)                | 0.0160   | 0.0023   | 0.9997   |
| KNN Regressor (Randomized Search CV)          | 0.0261   | 0.0088   | 0.9995   |
| SVR (Baseline)                                | 0.0043   | 0.0551   | 0.9999   |
| SVR (Bagging)                                 | 0.0058   | 0.0562   | 0.9999   |
| Linear Regression (Baseline)                  | 13.7068  | 0.8314   | 0.7354   |
| Linear Regression (Boosting)                  | 14.0666  | 0.7960   | 0.7285   |
| Linear Regression (Grid Search CV)            | 13.7068  | 0.8314   | 0.7354   |
| Linear Regression (Randomized Search CV)      | 13.7068  | 0.8314   | 0.7354   |
| Decision Tree Regressor (Baseline)            | 0.0018   | 0.0018   | 0.99996  |
| Decision Tree Regressor (Bagging/Random Forest)| 0.0758   | 0.0137   | 0.9985   |
| Decision Tree Regressor (Grid Search CV)      | 0.0180   | 0.0219   | 0.9997   |
| Decision Tree Regressor (Randomized Search CV)| 0.3678   | 0.0294   | 0.9929   |
| TensorFlow ANN Model                          | 0.0111   | 0.0245   | 0.9998   |


***Based on the evaluation comparison table, the KNN Regressor (Boosting) model appears to be the best performing. It boasts the highest R² Score of 0.999996 and the lowest Mean Squared Error (MSE) of 0.0002, along with a very low Mean Absolute Error (MAE) of 0.0009. These metrics indicate that it provides the most accurate predictions for FinalMandates among the models evaluated so far.***
