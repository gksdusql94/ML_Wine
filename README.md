# 🍷 Wine Quality Prediction Using KNN and Logistic Regression

This project explores machine learning techniques to classify wine quality as "good" or "bad" based on physicochemical properties. Two models, K-Nearest Neighbors (KNN) and Logistic Regression (LR), were used, evaluated, and optimized. The aim was to compare performance metrics like accuracy, F1-score, precision, and recall, and find the optimal model.

## 🎯 Objective

The objective of this project is to build and evaluate machine learning models that can accurately predict wine quality (good or bad) based on chemical features such as acidity, sugar content, pH, and alcohol levels.

## 📊 Dataset

The dataset used contains white wine quality measurements from the UCI Machine Learning Repository, consisting of over 4,000 entries. Features include acidity, sugar, pH, alcohol, and more. The target variable, quality, was binarized into "good" (quality > 5) and "bad" (quality <= 5).

## Models
### 1. **K-Nearest Neighbors (KNN)**:
   - Hyperparameters: k values, distance metrics (Euclidean, Manhattan), weights (uniform, distance).
   - The best KNN model used k=11, Manhattan distance, and distance-based weights.

```python
#values of for parameters of Knn model  is already given in the question, which are as follows
k_values = [1, 5, 9, 11]
distances = ["euclidean", "manhattan"]
weights = ["uniform", "distance"]
#creating an empty dataframe to populate the resultant values after the model performance
perf = pd.DataFrame(columns=['Experiment name', 'n_neighbors', 'distance', 'weights', 'Average F1'])
# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
#using for loop to go over all three parameters given in the questions
# for using sfold function defined in previous question
index_counter = 0
for k in k_values:
    for distance in distances:
        for weight in weights:
            # define the model with its parameters
              model_args = {'n_neighbors': k, 'weights': weight, 'metric': distance}

               # apply sfold cross validation here e are doing s=4, the same function used in last question
              expected_labels5, predicted_labels5, avg_error = sFold(5, train_data, train_labels, KNeighborsClassifier, model_args, f1_score)

              #calculate the average f1-score
              average_f1= f1_score(expected_labels5, predicted_labels5, average="weighted")

              performance_df_values = pd.DataFrame({'Experiment name': f'k={k}, distance={distance}, weights={weight}',
                                                  'n_neighbors': k,
                                                  'distance': distance,
                                                  'weights': weight,
                                                  'Average F1': average_f1},
                                                 index=[index_counter])


              perf = pd.concat([perf, performance_df_values])


df_sorted = perf.sort_values(by='Average F1', ascending=False)
df_sorted.reset_index(drop =True, inplace=True)
df_sorted
```

| Experiment name                           | k  | distance  | weights  | Average F1 |
|-------------------------------------------|----|-----------|----------|------------|
| k=11, distance=manhattan, weights=distance | 11 | manhattan | distance | 0.762225   |
| k=9, distance=manhattan, weights=distance  | 9  | manhattan | distance | 0.756261   |
| k=9, distance=euclidean, weights=distance  | 9  | euclidean | distance | 0.750974   |
| k=5, distance=manhattan, weights=distance  | 5  | manhattan | distance | 0.750917   |
| k=11, distance=euclidean, weights=distance | 11 | euclidean | distance | 0.749836   |
| k=5, distance=euclidean, weights=distance  | 5  | euclidean | distance | 0.740819   |
| k=1, distance=manhattan, weights=uniform   | 1  | manhattan | uniform  | 0.732666   |
| k=1, distance=manhattan, weights=distance  | 1  | manhattan | distance | 0.732666   |
| k=1, distance=euclidean, weights=uniform   | 1  | euclidean | uniform  | 0.729658   |
| k=1, distance=euclidean, weights=distance  | 1  | euclidean | distance | 0.729658   |
| k=5, distance=manhattan, weights=uniform   | 5  | manhattan | uniform  | 0.688057   |
| k=9, distance=manhattan, weights=uniform   | 9  | manhattan | uniform  | 0.686565   |
| k=11, distance=manhattan, weights=uniform  | 11 | manhattan | uniform  | 0.682463   |
| k=9, distance=euclidean, weights=uniform   | 9  | euclidean | uniform  | 0.680960   |
| k=11, distance=euclidean, weights=uniform  | 11 | euclidean | uniform  | 0.678459   |
| k=5, distance=euclidean, weights=uniform   | 5  | euclidean | uniform  | 0.678334   |



### 2. **Logistic Regression (LR)**:
   - Regularized LR tested with L1 and L2 penalties, and values of `C` (regularization strength).
   - The best LR model used L1 regularization with `C=10`.

```python
C_values = [ 0.1, 1.0, 10]
penalty_values = ['l1', 'l2']

## Creating an empty DataFrame for logistic regression performance
perform_df = pd.DataFrame(columns=['Experiment name', 'penalty', 'solver', 'C', 'Average F1'])

index_counter = 0

## Creating for loop to iterate over all the hyperparameters of logistic regression
for c in C_values:
    for p in penalty_values:
        # Define the model with its parameters
        solver = 'liblinear' if p == 'l1' else 'lbfgs'
        model_args = {'penalty': p, 'C': c, 'solver': solver, "max_iter":1000}

        ## Apply sFold cross-validation defined in question 17
        expected_labels4, predicted_labels4, avg_error = sFold(5, data, labels, LogisticRegression, model_args, f1_score)

        ## Calculate the average f1-score
        average_f1 = f1_score(expected_labels4, predicted_labels4, average="weighted")

        ## Populating the dataframe with values generated and the hyperparameters used
        perform_df_values = pd.DataFrame({'Experiment name': f'penalty={p}, solver={solver}, C={c}',
                                              'penalty': p,
                                              'solver': solver,
                                              'C': c,
                                              'Average F1': average_f1},
                                             index=[index_counter])

        ## Concatenating DataFrames
        perform_df = pd.concat([perform_df, perform_df_values])
## Display the resulting DataFrame
perform_df

# Sort the DataFrame by 'Average F1' in descending order
perform_df_sorted = perform_df.sort_values(by='Average F1', ascending=False).reset_index(drop=True)

# Display the sorted DataFrame
perform_df_sorted
```

![image](https://github.com/user-attachments/assets/df5b3474-73ee-488c-a9fc-3c78bbe2c752)


## 🧠 Results

The KNN model using Manhattan distance and inverse distance weighting (IDW) provided the best results with an F1 score of 0.87, outperforming Logistic Regression.

```python
# Best model parameters based on previous results: KNN
best_k = 11
best_distance = 'manhattan'
best_weights = 'distance'

# Re-train the model on full training data
knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, metric=best_distance)
knn.fit(train_data, train_labels)

# Make predictions on test data
test_pred = knn.predict(test_data)

# Calculate performance metrics
print("Precision: ", precision_score(test_labels, test_pred))
print("Recall: ", recall_score(test_labels, test_pred))
print("F1 Score: ", f1_score(test_labels, test_pred))
print("Confusion Matrix: ")
print(confusion_matrix(test_labels, test_pred))
print("Accuracy: ", accuracy_score(test_labels, test_pred))

# Calculate generalization error
gen_error = 1 - accuracy_score(test_labels, test_pred)
print("Generalization Error: ", gen_error)
```

```python
# Best model parameters based on previous results:IDW
best_penalty = 'l1'
best_C = 10.0
best_solver = 'liblinear'

# Re-train the model on full training data
lr = LogisticRegression(penalty=best_penalty, C=best_C, solver=best_solver, max_iter=1000)
lr.fit(train_data, train_labels)

# Make predictions on test data
test_pred = lr.predict(test_data)

# Calculate performance metrics
print("Precision: ", precision_score(test_labels, test_pred))
print("Recall: ", recall_score(test_labels, test_pred))
print("F1 Score: ", f1_score(test_labels, test_pred))
print("Confusion Matrix: ")
print(confusion_matrix(test_labels, test_pred))
print("Accuracy: ", accuracy_score(test_labels, test_pred))

# Calculate generalization error
gen_error = 1 - accuracy_score(test_labels, test_pred)
print("Generalization Error: ", gen_error)
```

| Model                    | F1 Score | Precision | Recall | Accuracy |
|--------------------------|----------|-----------|--------|----------|
| KNN (Manhattan, IDW)      | 0.87     | 0.86      | 0.89   | 0.83     |
| Logistic Regression (L2)  | 0.82     | 0.81      | 0.84   | 0.79     |

## Conclusion
KNN with k=11 and Manhattan distance outperformed logistic regression for this task. However, logistic regression performed competitively and may be preferred when model interpretability or faster predictions are necessary.

## Future Work
- Experiment with ensemble models such as Random Forest and Boosting.
- Further investigate feature engineering and hyperparameter tuning for potential improvements.

## ⚙️ Skills & Tools

- **Languages/Tools**: Python, Pandas, NumPy, Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Precision-recall analysis, ROC-AUC curves, cross-validation

## 💻 How to Run
Clone the repository:
```bash
git clone https://github.com/gksdusql94/ML_Wine.git
```


## 📈Visuals

### Pair Plot of Features vs. Quality

```python
import seaborn as sns
sns.pairplot(df, hue = 'quality') # I want to compare all files with quality
plt.show()
```
![image](https://github.com/user-attachments/assets/c9072b96-6bca-40f4-a3cc-ca9a87769a59)


### ROC Curve for KNN Model

```python
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for KNN')
    plt.legend(loc="lower right")
    plt.show()

    # Find optimal threshold
    f1_scores = [f1_score(test_labels, [1 if x >= t else 0 for x in knn_probs]) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold for KNN:", optimal_threshold)

    # Predict labels using the optimal threshold
    optimal_pred = [1 if x >= optimal_threshold else 0 for x in knn_probs]

    # Print F1 score using the optimal threshold
    print("F1 Score using Optimal Threshold:", f1_score(test_labels, optimal_pred))

except Exception as e:
    print("Error:", e)
```
![image](https://github.com/user-attachments/assets/1e437d29-05fb-4a11-a25c-abb69439f7d0)


