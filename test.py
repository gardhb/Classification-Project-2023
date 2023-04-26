import pandas as pd
import numpy as np

data = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')
selected_features = data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre', 'Type']]

# Filter rows based on the 'Type' column
training_data = selected_features[selected_features['Type'] == 'Train']
test_data = selected_features[selected_features['Type'] == 'Test']

# Separate features and labels for both training and test datasets
X_train = training_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']]
y_train = training_data['Genre']

X_test = test_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']]
y_test = test_data['Genre']

# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
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
