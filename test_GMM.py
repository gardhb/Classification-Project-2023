from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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

# Identify unique class labels
unique_classes = np.unique(y_train)

# Train a GMM for each class
gmms = {}
for cls in unique_classes:
    X_train_cls = X_train[y_train == cls]
    gmm = GaussianMixture(n_components=2, random_state=42)  # You can try different values for n_components
    gmm.fit(X_train_cls)
    gmms[cls] = gmm

# Classify test instances based on the highest posterior probability
y_pred = []
for x in X_test.values:  # Convert the DataFrame to a NumPy array
    posteriors = [gmm.score_samples(x.reshape(1, -1)) for gmm in gmms.values()]
    predicted_class = unique_classes[np.argmax(posteriors)]
    y_pred.append(predicted_class)

# Evaluate the classifier
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix
classes = np.unique(y_train)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

