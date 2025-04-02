# EY Open AI Data Science Challenge - Urban Heat Island (UHI) Index

This repository contains the solution for the **EY Open AI Data Science Challenge**, focusing on predicting the Urban Heat Island (UHI) Index using various machine learning models.

![image](https://github.com/user-attachments/assets/ecceaac4-63b9-4139-bce6-e0dc0ad4d466)


## Dataset Features
Key features used in model training include:
- **lwir11**
- **Avg Wind Speed [m/s]**
- **B01, B09, NBR2**
- **Solar Flux [W/m^2]**
- **SAVI**

## Model Performance

| Model                 | Hyperparameters                                      | Most Important Feature(s) | Train R² | Val R² |
|-----------------------|------------------------------------------------|---------------------------------------------|---------|--------|
| Linear Regression     | -                                              | -                                           | 0.1962  | 0.2347 |
| Ridge Regression      | {'alpha': 1}                                  | -                                           | 0.1962  | 0.2351 |
| Lasso Regression      | {'alpha': 0.1}                                | -                                           | 0.0000  | -0.0002 |
| Decision Tree        | {'min_samples_split': 10, 'max_depth': 10}     | lwir11, Wind Speed, B01, Solar Flux        | 0.5642  | 0.4357 |
| Random Forest        | {'n_estimators': 200, 'max_depth': None}       | lwir11, B01, Wind Speed, B09, Solar Flux   | 0.9664  | 0.7635 |
| SVM                  | {'kernel': 'linear', 'C': 0.1}                 | -                                           | -0.0041 | -0.006 |
| Elastic Net          | {'l1_ratio': 0.5, 'alpha': 10}                 | -                                           | 0.0000  | -0.0002 |
| Huber Regression     | {'alpha': 0.0001}                              | -                                           | 0.1934  | 0.2358 |
| Bayesian Ridge       | {'lambda_1': 1e-06, 'alpha_1': 1e-06}          | -                                           | 0.1929  | 0.2328 |
| Poisson Regressor    | {'alpha': 0.01}                                | -                                           | 0.1864  | 0.2277 |
| Quantile Regressor   | {'alpha': 0.0001}                              | -                                           | 0.1862  | 0.2344 |
| MLP (Neural Net)     | {'hidden_layer_sizes': (50, 50), 'alpha': 0.01}| -                                           | -2.4288 | -4.0363 |
| ADA Boost            | {'n_estimators': 200, 'learning_rate': 0.5}    | lwir11, Wind Speed, B01                     | 0.2312  | 0.2422 |
| Gradient Boosting    | {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1} | lwir11, B01, B09, Solar Flux, Wind Speed | 0.9355  | 0.7522 |
| XGBoost              | {'n_estimators': 200, 'learning_rate': 0.2}    | SAVI, lwir11, Wind Speed                    | 0.9717  | 0.7599 |
| LightGBM             | {'n_estimators': 200, 'learning_rate': 0.2}    | B01, lwir11, B09, Solar Flux, NBR2          | 0.9518  | 0.7457 |
| CatBoost             | {'learning_rate': 0.2, 'iterations': 200}      | lwir11, Wind Speed, B01, Solar Flux         | 0.7791  | 0.6344 |
| KNN Regressor        | {'weights': 'distance', 'n_neighbors': 5, 'metric': 'manhattan'} | -       | 0.9981  | 0.5138 |

### Ensemble Methods

| Model Combination               | Method                               | Train R² | Val R² |
|----------------------------------|--------------------------------------|----------|--------|
| Top 6 Models                     | Voting Classifier                    | 0.9599   | 0.7477 |
| Top 6 Models                     | Stacking                             | 0.9668   | 0.7664 |
| Top 6 Models                     | Stacking (XGBRegressor as meta)      | 0.9849   | 0.8110 |

## Final Leaderboard Score
- **Participant**: Diwakar Sehgal
- **Leaderboard Score**: **0.5278**


## Conclusion
The best-performing model was the **Stacking Ensemble with XGBRegressor as the meta learner**, achieving an **R² of 0.811 on validation data**. Future work could include hyperparameter tuning, additional feature engineering, and further ensemble experimentation.

