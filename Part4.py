import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

k = 11

# Read both datasets
data_5s = pd.read_csv('Classification music/GenreClassData_5s.txt', sep='\t')
data_10s = pd.read_csv('Classification music/GenreClassData_10s.txt', sep='\t')
data_30s = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')

# Concatenate the datasets
combined_data = pd.concat([data_5s,data_10s, data_30s], ignore_index=True)

#selected_features = data_30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre', 'Type']]

selected_features = combined_data.drop('tempo', axis=1)
# Filter rows based on the 'Type' column
training_data = selected_features[selected_features['Type'] == 'Train']
test_data = selected_features[selected_features['Type'] == 'Test']

# Separate features and labels for both training and test datasets
X_train = training_data.drop(['Genre', 'Type','Track ID', 'File','GenreID'], axis=1)
y_train = training_data['Genre']

X_test = test_data.drop(['Genre', 'Type','Track ID', 'File','GenreID'], axis=1)
y_test = test_data['Genre']

# Convert datasets into numpy
X_train_NP = X_train.values
y_train_NP = y_train.values

X_test_NP = X_test.values
y_test_NP = y_test.values

# Compute vector of distances
distances = []
i = 0
while i < len(X_test_NP):
    distances.append(np.linalg.norm(X_train_NP - X_test_NP[i], axis=1))
    i = i+1

# Find kNN
k_nearest_neighbor_id = []
i = 0
while i < len(distances):
    k_nearest_neighbor_id.append(distances[i].argsort()[:k])
    i = i+1

# Connect values with genre
k_nearest_neighbor_genre = []
i = 0
while i < len(k_nearest_neighbor_id):
    k_nearest_neighbor_genre.append(y_train_NP[k_nearest_neighbor_id[i]])
    i = i+1

k_nearest_neighbor_genre_df = pd.DataFrame(k_nearest_neighbor_genre)

# Make a prediction of genre
#genre_prediction = []
#i = 0
#while i < len(k_nearest_neighbor_genre):
#    genre_prediction.append(ss.mode(k_nearest_neighbor_genre[i])[0])
#    i = i+1

genre_prediction = []
i = 0
while i < len(k_nearest_neighbor_genre_df):
    genre_prediction.append(k_nearest_neighbor_genre_df.iloc[i].mode().values[0])
    i = i+1

# Create confusion matrix
i = 0
classes = np.unique(y_train_NP)
confusion_matrix = np.zeros((len(classes), len(classes)))
for i in range(len(classes)):
    for j in range(len(classes)):
      confusion_matrix[i, j] = np.sum((y_test_NP == classes[i]) & (np.array(genre_prediction).reshape((len(genre_prediction))) == classes[j]))

# Print confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

# Compress confusion matrix to find TP, TN, FP and FN for each class
i = 0
j = 0
tp_tn_fp_fn_matrix = np.zeros((len(confusion_matrix[0]), 4))
for i in range(len(confusion_matrix[0])):
    for j in range(len(confusion_matrix[0])):
        if i == j:
            tp_tn_fp_fn_matrix[i][0] = confusion_matrix[i][j]
        else:
            tp_tn_fp_fn_matrix[i][3] = tp_tn_fp_fn_matrix[i][3] + confusion_matrix[i][j]
            tp_tn_fp_fn_matrix[j][2] = tp_tn_fp_fn_matrix[i][2] + confusion_matrix[i][j]
        
sum_confusion_matrix = sum(list(map(sum, confusion_matrix)))
tp_tn_fp_fn_matrix[:, 1] = sum_confusion_matrix - (tp_tn_fp_fn_matrix[:, 3] + tp_tn_fp_fn_matrix[:, 2] + tp_tn_fp_fn_matrix[:, 0])

# Find TP, TN, FP and FN for the whole system
i = 0
system_tp_tn_fp_fn_matrix = np.zeros(len(tp_tn_fp_fn_matrix[0]))
for i in range(len(tp_tn_fp_fn_matrix[0])):
    system_tp_tn_fp_fn_matrix[i] = sum(tp_tn_fp_fn_matrix[:, i])

# Calculate accuracy and error rate
result_accuracy = system_tp_tn_fp_fn_matrix[0]/sum_confusion_matrix
result_error_rate = 1 - result_accuracy

print("Accuracy: ", result_accuracy)
print("Error rate: ", result_error_rate)