import pandas as pd
import numpy as np

# Read both datasets
data_5s = pd.read_csv('Classification music/GenreClassData_5s.txt', sep='\t')
data_10s = pd.read_csv('Classification music/GenreClassData_10s.txt', sep='\t')
data_30s = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')

# Concatenate the datasets
combined_data = pd.concat([data_5s,data_10s, data_30s], ignore_index=True)

selected_features = combined_data.drop('tempo', axis=1)
# Filter rows based on the 'Type' column
training_data = selected_features[selected_features['Type'] == 'Train']
test_data = selected_features[selected_features['Type'] == 'Test']

# Separate features and labels for both training and test datasets
X_train = training_data.drop(['Genre', 'Type','Track ID', 'File','GenreID'], axis=1)
y_train = training_data['Genre']

X_test = test_data.drop(['Genre', 'Type','Track ID', 'File','GenreID'], axis=1)
y_test = test_data['Genre']

# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
