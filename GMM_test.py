
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from GMM import GaussianMixtureModel

data_5s = pd.read_csv('Classification music/GenreClassData_5s.txt', sep='\t')
data_10s = pd.read_csv('Classification music/GenreClassData_10s.txt', sep='\t')
data_30s = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')

combined_data = pd.concat([data_5s,data_10s, data_30s], ignore_index=True)
# Filter rows based on the 'Type' column
selected_features = combined_data.drop('tempo', axis=1)

training_data = selected_features[selected_features['Type'] == 'Train']
test_data = selected_features[selected_features['Type'] == 'Test']

X_train = training_data.drop(['Genre', 'Type','Track ID', 'File','GenreID'], axis=1)
y_train = training_data['GenreID']

X_test = test_data.drop(['Genre', 'Type','Track ID', 'File','GenreID'], axis=1)
y_test = test_data['GenreID']

# Convert datasets into numpy
X_train_NP = X_train.values
y_train_NP = y_train.values

X_test_NP = X_test.values
y_test_NP = y_test.values

# Create an instance of the GaussianMixtureModel class
gmm = GaussianMixtureModel(n_components=3, max_iter=100, tol=1e-3)

# Fit the model to the training data
gmm.fit(X_train_NP)

# Make predictions using the test data
y_pred = gmm.predict(X_test_NP)

# Evaluate the performance
print(classification_report(y_test, y_pred))