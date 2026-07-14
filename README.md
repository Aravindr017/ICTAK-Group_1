# Data Preprocessing - Election Result Portugal 2019
The task which were doing as the group during the course at ICTAK TVM.

Things should be included in this :
  - source of the dataset
  - data understanding inferences
  - the steps we are taking
    
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# Outlier Handling - Election Result Portugal 2019
   - first we have identified  the outlier using Boxplot.
   - after finding outliers , we try to clip with IQR method
   - finally ,again verify with Boxplot

######################################################################################################################################################################


### . Correlation Analysis
- Computed the Pearson correlation matrix for all numerical features.
- Identified relationships between variables.
### . Correlation Heatmap
- Visualized the correlation matrix using a heatmap.
- Used the heatmap to identify highly correlated features.
### . Feature Selection using Correlation
- Removed highly correlated features using a correlation threshold of **0.90**.
- This helps reduce multicollinearity and improves model performance.
### . Mutual Information
- Calculated Mutual Information scores between each feature and the target variable (`FinalMandates`).
- Ranked features according to their importance.
### . Removing Low-Importance Features
- Removed features with very low Mutual Information scores.
- Retained only informative features for model training.
- ### . Feature Scaling
- Applied **StandardScaler** to numerical features.
- Standardized features to have:
  - Mean = 0
  - Standard Deviation = 1
- Compared feature values before and after scaling.
###########################################################################################################################################################################3
