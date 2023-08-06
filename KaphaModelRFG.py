import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset_filepath = '/content/drive/My Drive/av/Kapha_Dataset.csv'
df = pd.read_csv(dataset_filepath)

X = df[['MeanBMI', 'SedentaryMinutes', 'LightlyActiveMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes']]
y = df['Kapha_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest Regressor Training
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error on the test set:", rmse)
print("R-squared (R2) score on the test set:", r2)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual Kapha Score')
plt.ylabel('Predicted Kapha Score')
plt.title('Actual vs. Predicted Kapha Score')
plt.savefig('/content/drive/My Drive/av/actual_vs_predicted.png')  # Save the plot
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.savefig('/content/drive/My Drive/av/residual_distribution.png')  # Save the plot
plt.show()

feature_importance = pd.Series(rf_regressor.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Kapha Score Prediction')
plt.savefig('/content/drive/My Drive/av/feature_importance.png')  # Save the plot
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Kapha_Score'], kde=True)
plt.xlabel('Kapha Score')
plt.ylabel('Frequency')
plt.title('Distribution of Kapha Score')
plt.savefig('/content/drive/My Drive/av/kapha_score_distribution.png')  # Save the plot
plt.show()

sns.pairplot(df, vars=X.columns, diag_kind='kde', hue='Kapha_Score')
plt.suptitle('Pairplot of Features with Kapha Score')
plt.savefig('/content/drive/My Drive/av/pairplot.png')  # Save the plot
plt.show()
