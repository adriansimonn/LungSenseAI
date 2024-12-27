# LungSense.AI
# Author: Adrian Simon
# MODEL FOR LUNG CANCER PREDICTION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('dataset.csv')

# Note that the last column (Y) is the target column
# Last column is 3000 NOs and YESs representing the presence of lung cancer
# Note that the first 15 columns (X) contain input results from data from 3000 people
# Presence of lung cancer is a function of the first 15 columns (user provided factors)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splits data set into training and testing data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling of features to normalize data for model training, avoids bias towards
# certain features with larger ranges of values (in this instance, the age column)
# fit_transform() calculates and applies scaling, transform() applies scaling without
# recalculation, (uses calculation from last call to fit_transform())
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
# After extensive testing of different models including Random Forest (91.5%), XGBoost (93.333%),
# Logistic Regression (96.5%), and K-Nearest Neighbors (92.5%), Logistic Regression was chosen
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = 100 * accuracy_score(y_test, prediction)

# Evaluation of model
print(f'Accuracy: {accuracy}')

# Saving the model
import joblib
joblib.dump(model, 'lungSenseModel.pkl')
joblib.dump(scaler, 'scaler.pkl')