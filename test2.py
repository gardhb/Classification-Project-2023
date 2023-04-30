#import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix


k = 5

data_30 = pd.read_csv('Classification music/GenreClassData_30s.txt', sep="\t")

selected_features = data_30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo', 'Genre', 'Type']]

# Filter rows based on the 'Type' column
training_data = selected_features[selected_features['Type'] == 'Train']
test_data = selected_features[selected_features['Type'] == 'Test']

# Separate features and labels for both training and test datasets
X_train = training_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']]
y_train = training_data['Genre']

X_test = test_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']]
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


# Separate features into genres
features_pop = training_data[training_data['Genre'] == 'pop']
features_disco = training_data[training_data['Genre'] == 'disco']
features_metal = training_data[training_data['Genre'] == 'metal']
features_classical = training_data[training_data['Genre'] == 'classical']

# Plot histograms

kwargs = dict(alpha=0.5, bins=10)
plt.hist(features_pop['spectral_rolloff_mean'], **kwargs, color='g', label='Pop')
plt.hist(features_disco['spectral_rolloff_mean'], **kwargs, color='r', label='Disco')
plt.hist(features_metal['spectral_rolloff_mean'], **kwargs, color='y', label='Metal')
plt.hist(features_classical['spectral_rolloff_mean'], **kwargs, color='b', label='Classical')
plt.gca().set(title='Histogram of Spectral Rolloff Mean for pop, disco, metal and classical samples', ylabel='Spectral rolloff mean')
plt.legend()
plt.show()

kwargs = dict(alpha=0.5, bins=100)
plt.hist(features_pop['mfcc_1_mean'], **kwargs, color='g', label='Pop')
plt.hist(features_disco['mfcc_1_mean'], **kwargs, color='r', label='Disco')
plt.hist(features_metal['mfcc_1_mean'], **kwargs, color='y', label='Metal')
plt.hist(features_classical['mfcc_1_mean'], **kwargs, color='b', label='Classical')
plt.gca().set(title='Histogram of MFCC 1 mean for pop, disco, metal and classical samples', ylabel='MFCC 1 mean')
plt.legend()
plt.show()

kwargs = dict(alpha=0.5, bins=100)
plt.hist(features_pop['spectral_centroid_mean'], **kwargs, color='g', label='Pop')
plt.hist(features_disco['spectral_centroid_mean'], **kwargs, color='r', label='Disco')
plt.hist(features_metal['spectral_centroid_mean'], **kwargs, color='y', label='Metal')
plt.hist(features_classical['spectral_centroid_mean'], **kwargs, color='b', label='Classical')
plt.gca().set(title='Histogram of Spectral Centroid Mean for pop, disco, metal and classical samples', ylabel='Spectral centroid mean')
plt.legend()
plt.show()

kwargs = dict(alpha=0.5, bins=100)
plt.hist(features_pop['tempo'], **kwargs, color='g', label='Pop')
plt.hist(features_disco['tempo'], **kwargs, color='r', label='Disco')
plt.hist(features_metal['tempo'], **kwargs, color='y', label='Metal')
plt.hist(features_classical['tempo'], **kwargs, color='b', label='Classical')
plt.gca().set(title='Histogram of Tempo for pop, disco, metal and classical samples', ylabel='Tempo')
plt.legend()
plt.show()