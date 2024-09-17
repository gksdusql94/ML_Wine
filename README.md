# Wine Quality Prediction Using KNN and Logistic Regression

This project focuses on predicting wine quality based on various chemical properties using machine learning models. We implemented two models: K-Nearest Neighbors (KNN) and Logistic Regression, comparing their performance and optimizing hyperparameters to achieve the best results.

## Objective

The objective of this project is to build and evaluate machine learning models that can accurately predict wine quality (good or bad) based on chemical features such as acidity, sugar content, pH, and alcohol levels.

## Dataset

The dataset used in this project is a real-world dataset of wine quality, consisting of over 4,000 entries and 11 features, including:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target variable is wine quality, which is classified into two categories:
- **Good quality**: wine quality > 5
- **Bad quality**: wine quality ≤ 5

## Key Achievements

1. **Data Preprocessing**: Cleaned and standardized the dataset for better model performance. Removed redundant features based on correlation analysis.
2. **Model Implementation**:
   - Developed custom KNN and Logistic Regression models.
   - Experimented with different hyperparameters such as distance metrics (Euclidean/Manhattan) and regularization types (L1/L2).
3. **Model Evaluation**: 
   - Used cross-validation to evaluate the models.
   - Achieved the best F1 score of 0.87 with the KNN model using Manhattan distance and inverse distance weighting.
4. **Performance Improvement**:
   - Standardized the dataset to enhance model accuracy.
   - The KNN model outperformed Logistic Regression, with a performance improvement of 10% after standardization.

## Results

The KNN model with Manhattan distance and inverse distance weighting proved to be the most effective, achieving an F1 score of 0.87. Standardizing the data significantly improved model performance, especially for KNN, which outperformed Logistic Regression on this dataset.

| Model                    | F1 Score | Precision | Recall | Accuracy |
|--------------------------|----------|-----------|--------|----------|
| KNN (Manhattan, IDW)      | 0.87     | 0.86      | 0.89   | 0.83     |
| Logistic Regression (L2)  | 0.82     | 0.81      | 0.84   | 0.79     |

## Skills & Tools

- **Languages/Tools**: Python, Pandas, NumPy, Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Precision-recall analysis, ROC-AUC curves, cross-validation

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wine-quality-prediction.git