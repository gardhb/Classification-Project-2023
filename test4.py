import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_5s = pd.read_csv('Classification music/GenreClassData_5s.txt', sep='\t')
data_10s = pd.read_csv('Classification music/GenreClassData_10s.txt', sep='\t')
data_30s = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')

combined_data = pd.concat([data_5s,data_10s, data_30s], ignore_index=True)
# Filter rows based on the 'Type' column
selected_features = combined_data.drop('tempo', axis=1)

training_data = selected_features[selected_features['Type'] == 'Train']
test_data = selected_features[selected_features['Type'] == 'Test']

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix
classes = np.unique(y_train)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))