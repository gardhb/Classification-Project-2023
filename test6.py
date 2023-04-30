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

# Make a prediction of genre
genre_prediction = []
i = 0
while i < len(k_nearest_neighbor_genre):
    genre_prediction.append(ss.mode(k_nearest_neighbor_genre[i])[0])
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
kwargs = dict(alpha=0.5, bins=100)
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

ax0.hist(features_pop['spectral_rolloff_mean'], **kwargs, color='g', label='Pop')
ax0.hist(features_disco['spectral_rolloff_mean'], **kwargs, color='r', label='Disco')
ax0.hist(features_metal['spectral_rolloff_mean'], **kwargs, color='y', label='Metal')
ax0.hist(features_classical['spectral_rolloff_mean'], **kwargs, color='b', label='Classical')
ax0.set_ylabel('Spectral rolloff mean')
ax0.legend();

ax1.hist(features_pop['mfcc_1_mean'], **kwargs, color='g', label='Pop')
ax1.hist(features_disco['mfcc_1_mean'], **kwargs, color='r', label='Disco')
ax1.hist(features_metal['mfcc_1_mean'], **kwargs, color='y', label='Metal')
ax1.hist(features_classical['mfcc_1_mean'], **kwargs, color='b', label='Classical')
ax1.set_ylabel('MFCC 1 mean')
ax1.legend();

ax2.hist(features_pop['spectral_centroid_mean'], **kwargs, color='g', label='Pop')
ax2.hist(features_disco['spectral_centroid_mean'], **kwargs, color='r', label='Disco')
ax2.hist(features_metal['spectral_centroid_mean'], **kwargs, color='y', label='Metal')
ax2.hist(features_classical['spectral_centroid_mean'], **kwargs, color='b', label='Classical')
ax2.set_ylabel('Spectral centroid mean')
ax2.legend();

ax3.hist(features_pop['tempo'], **kwargs, color='g', label='Pop')
ax3.hist(features_disco['tempo'], **kwargs, color='r', label='Disco')
ax3.hist(features_metal['tempo'], **kwargs, color='y', label='Metal')
ax3.hist(features_classical['tempo'], **kwargs, color='b', label='Classical')
ax3.set_ylabel('Tempo')
ax3.legend();


#---------------------------------------------------------------------------

# Separate features and labels for both training and test datasets
X_train_2 = training_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean']]

X_test_2 = test_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean']]

# Convert datasets into numpy
X_train_NP_2 = X_train_2.values

X_test_NP_2 = X_test_2.values

# Compute vector of distances
distances_2 = []
i = 0
while i < len(X_test_NP_2):
    distances_2.append(np.linalg.norm(X_train_NP_2 - X_test_NP_2[i], axis=1))
    i = i+1

# Find kNN
k_nearest_neighbor_id_2 = []
i = 0
while i < len(distances_2):
    k_nearest_neighbor_id_2.append(distances_2[i].argsort()[:k])
    i = i+1

# Connect values with genre
k_nearest_neighbor_genre_2 = []
i = 0
while i < len(k_nearest_neighbor_id_2):
    k_nearest_neighbor_genre_2.append(y_train_NP[k_nearest_neighbor_id_2[i]])
    i = i+1

# Make a prediction of genre
genre_prediction_2 = []
i = 0
while i < len(k_nearest_neighbor_genre_2):
    genre_prediction_2.append(ss.mode(k_nearest_neighbor_genre_2[i])[0])
    i = i+1

# Create confusion matrix
i = 0
confusion_matrix_2 = np.zeros((len(classes), len(classes)))
for i in range(len(classes)):
    for j in range(len(classes)):
      confusion_matrix_2[i, j] = np.sum((y_test_NP == classes[i]) & (np.array(genre_prediction_2).reshape((len(genre_prediction_2))) == classes[j]))

# Compress confusion matrix to find TP, TN, FP and FN for each class
i = 0
j = 0
tp_tn_fp_fn_matrix_2 = np.zeros((len(confusion_matrix_2[0]), 4))
for i in range(len(confusion_matrix_2[0])):
    for j in range(len(confusion_matrix_2[0])):
        if i == j:
            tp_tn_fp_fn_matrix_2[i][0] = confusion_matrix_2[i][j]
        else:
            tp_tn_fp_fn_matrix_2[i][3] = tp_tn_fp_fn_matrix_2[i][3] + confusion_matrix_2[i][j]
            tp_tn_fp_fn_matrix_2[j][2] = tp_tn_fp_fn_matrix_2[i][2] + confusion_matrix_2[i][j]
        
sum_confusion_matrix_2 = sum(list(map(sum, confusion_matrix_2)))
tp_tn_fp_fn_matrix_2[:, 1] = sum_confusion_matrix_2 - (tp_tn_fp_fn_matrix_2[:, 3] + tp_tn_fp_fn_matrix_2[:, 2] + tp_tn_fp_fn_matrix_2[:, 0])

# Find TP, TN, FP and FN for the whole system
i = 0
system_tp_tn_fp_fn_matrix_2 = np.zeros(len(tp_tn_fp_fn_matrix_2[0]))
for i in range(len(tp_tn_fp_fn_matrix_2[0])):
    system_tp_tn_fp_fn_matrix_2[i] = sum(tp_tn_fp_fn_matrix_2[:, i])

# Calculate accuracy and error rate
result_accuracy_2 = system_tp_tn_fp_fn_matrix_2[0]/sum_confusion_matrix_2
result_error_rate_2 = 1 - result_accuracy_2

# Test some of the remaining features on pop, disco, metal and classical genres
# Set up 
unselected_features = data_30[['zero_cross_rate_mean', 'zero_cross_rate_std', 'rmse_mean', 'rmse_var', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'spectral_rolloff_var', 'spectral_contrast_mean', 'spectral_contrast_var', 'spectral_flatness_mean', 'spectral_flatness_var', 'chroma_stft_1_mean', 'chroma_stft_1_std','mfcc_12_mean', 'mfcc_1_std', 'Genre', 'Type']]
unused_training_data = unselected_features[unselected_features['Type'] == 'Train']

# Separate features into genres
unused_features_pop = unused_training_data[unused_training_data['Genre'] == 'pop']
unused_features_disco = unused_training_data[unused_training_data['Genre'] == 'disco']
unused_features_metal = unused_training_data[unused_training_data['Genre'] == 'metal']
unused_features_classical = unused_training_data[unused_training_data['Genre'] == 'classical']

# Plot histograms
fig_2_0, ((ay0, ay1), (ay2, ay3), (ay4, ay5), (ay6, ay7)) = plt.subplots(nrows=4, ncols=2)

ay0.hist(unused_features_pop['zero_cross_rate_mean'], **kwargs, color='g', label='Pop')
ay0.hist(unused_features_disco['zero_cross_rate_mean'], **kwargs, color='r', label='Disco')
ay0.hist(unused_features_metal['zero_cross_rate_mean'], **kwargs, color='y', label='Metal')
ay0.hist(unused_features_classical['zero_cross_rate_mean'], **kwargs, color='b', label='Classical')
ay0.set_ylabel('zero_cross_rate_mean')
ay0.legend();

ay1.hist(unused_features_pop['zero_cross_rate_std'], **kwargs, color='g', label='Pop')
ay1.hist(unused_features_disco['zero_cross_rate_std'], **kwargs, color='r', label='Disco')
ay1.hist(unused_features_metal['zero_cross_rate_std'], **kwargs, color='y', label='Metal')
ay1.hist(unused_features_classical['zero_cross_rate_std'], **kwargs, color='b', label='Classical')
ay1.set_ylabel('zero_cross_rate_std')
ay1.legend();

ay2.hist(unused_features_pop['rmse_mean'], **kwargs, color='g', label='Pop')
ay2.hist(unused_features_disco['rmse_mean'], **kwargs, color='r', label='Disco')
ay2.hist(unused_features_metal['rmse_mean'], **kwargs, color='y', label='Metal')
ay2.hist(unused_features_classical['rmse_mean'], **kwargs, color='b', label='Classical')
ay2.set_ylabel('rmse_mean')
ay2.legend();

ay3.hist(unused_features_pop['rmse_var'], **kwargs, color='g', label='Pop')
ay3.hist(unused_features_disco['rmse_var'], **kwargs, color='r', label='Disco')
ay3.hist(unused_features_metal['rmse_var'], **kwargs, color='y', label='Metal')
ay3.hist(unused_features_classical['rmse_var'], **kwargs, color='b', label='Classical')
ay3.set_ylabel('rmse_var')
ay3.legend();

ay4.hist(unused_features_pop['spectral_centroid_var'], **kwargs, color='g', label='Pop')
ay4.hist(unused_features_disco['spectral_centroid_var'], **kwargs, color='r', label='Disco')
ay4.hist(unused_features_metal['spectral_centroid_var'], **kwargs, color='y', label='Metal')
ay4.hist(unused_features_classical['spectral_centroid_var'], **kwargs, color='b', label='Classical')
ay4.set_ylabel('spectral_centroid_var')
ay4.legend();

ay5.hist(unused_features_pop['spectral_bandwidth_mean'], **kwargs, color='g', label='Pop')
ay5.hist(unused_features_disco['spectral_bandwidth_mean'], **kwargs, color='r', label='Disco')
ay5.hist(unused_features_metal['spectral_bandwidth_mean'], **kwargs, color='y', label='Metal')
ay5.hist(unused_features_classical['spectral_bandwidth_mean'], **kwargs, color='b', label='Classical')
ay5.set_ylabel('spectral_bandwidth_mean')
ay5.legend();

ay6.hist(unused_features_pop['spectral_bandwidth_var'], **kwargs, color='g', label='Pop')
ay6.hist(unused_features_disco['spectral_bandwidth_var'], **kwargs, color='r', label='Disco')
ay6.hist(unused_features_metal['spectral_bandwidth_var'], **kwargs, color='y', label='Metal')
ay6.hist(unused_features_classical['spectral_bandwidth_var'], **kwargs, color='b', label='Classical')
ay6.set_ylabel('spectral_bandwidth_var')
ay6.legend();

ay7.hist(unused_features_pop['spectral_rolloff_var'], **kwargs, color='g', label='Pop')
ay7.hist(unused_features_disco['spectral_rolloff_var'], **kwargs, color='r', label='Disco')
ay7.hist(unused_features_metal['spectral_rolloff_var'], **kwargs, color='y', label='Metal')
ay7.hist(unused_features_classical['spectral_rolloff_var'], **kwargs, color='b', label='Classical')
ay7.set_ylabel('spectral_rolloff_var')
ay7.legend();

fig_2_1, ((az0, az1), (az2, az3), (az4, az5), (az6, az7)) = plt.subplots(nrows=4, ncols=2)
fig_2_1.tight_layout(pad=5.0)

az0.hist(unused_features_pop['spectral_contrast_mean'], **kwargs, color='g', label='Pop')
az0.hist(unused_features_disco['spectral_contrast_mean'], **kwargs, color='r', label='Disco')
az0.hist(unused_features_metal['spectral_contrast_mean'], **kwargs, color='y', label='Metal')
az0.hist(unused_features_classical['spectral_contrast_mean'], **kwargs, color='b', label='Classical')
az0.set_ylabel('spectral_contrast_mean')
az0.legend();

az1.hist(unused_features_pop['spectral_contrast_var'], **kwargs, color='g', label='Pop')
az1.hist(unused_features_disco['spectral_contrast_var'], **kwargs, color='r', label='Disco')
az1.hist(unused_features_metal['spectral_contrast_var'], **kwargs, color='y', label='Metal')
az1.hist(unused_features_classical['spectral_contrast_var'], **kwargs, color='b', label='Classical')
az1.set_ylabel('spectral_contrast_var')
az1.legend();

az2.hist(unused_features_pop['spectral_flatness_mean'], **kwargs, color='g', label='Pop')
az2.hist(unused_features_disco['spectral_flatness_mean'], **kwargs, color='r', label='Disco')
az2.hist(unused_features_metal['spectral_flatness_mean'], **kwargs, color='y', label='Metal')
az2.hist(unused_features_classical['spectral_flatness_mean'], **kwargs, color='b', label='Classical')
az2.set_ylabel('spectral_flatness_mean')
az2.legend();

az3.hist(unused_features_pop['spectral_flatness_var'], **kwargs, color='g', label='Pop')
az3.hist(unused_features_disco['spectral_flatness_var'], **kwargs, color='r', label='Disco')
az3.hist(unused_features_metal['spectral_flatness_var'], **kwargs, color='y', label='Metal')
az3.hist(unused_features_classical['spectral_flatness_var'], **kwargs, color='b', label='Classical')
az3.set_ylabel('spectral_flatness_var')
az3.legend();

az4.hist(unused_features_pop['chroma_stft_1_mean'], **kwargs, color='g', label='Pop')
az4.hist(unused_features_disco['chroma_stft_1_mean'], **kwargs, color='r', label='Disco')
az4.hist(unused_features_metal['chroma_stft_1_mean'], **kwargs, color='y', label='Metal')
az4.hist(unused_features_classical['chroma_stft_1_mean'], **kwargs, color='b', label='Classical')
az4.set_ylabel('chroma_stft_1_mean')
az4.legend();

az5.hist(unused_features_pop['chroma_stft_1_std'], **kwargs, color='g', label='Pop')
az5.hist(unused_features_disco['chroma_stft_1_std'], **kwargs, color='r', label='Disco')
az5.hist(unused_features_metal['chroma_stft_1_std'], **kwargs, color='y', label='Metal')
az5.hist(unused_features_classical['chroma_stft_1_std'], **kwargs, color='b', label='Classical')
az5.set_ylabel('chroma_stft_1_std')
az5.legend();

az6.hist(unused_features_pop['mfcc_12_mean'], **kwargs, color='g', label='Pop')
az6.hist(unused_features_disco['mfcc_12_mean'], **kwargs, color='r', label='Disco')
az6.hist(unused_features_metal['mfcc_12_mean'], **kwargs, color='y', label='Metal')
az6.hist(unused_features_classical['mfcc_12_mean'], **kwargs, color='b', label='Classical')
az6.set_ylabel('mfcc_12_mean')
az6.legend();

az7.hist(unused_features_pop['mfcc_1_std'], **kwargs, color='g', label='Pop')
az7.hist(unused_features_disco['mfcc_1_std'], **kwargs, color='r', label='Disco')
az7.hist(unused_features_metal['mfcc_1_std'], **kwargs, color='y', label='Metal')
az7.hist(unused_features_classical['mfcc_1_std'], **kwargs, color='b', label='Classical')
az7.set_ylabel('mfcc_1_std')
az7.legend();

#----------------------------------------------------------------------------

selected_features_3 = data_30[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'spectral_rolloff_var', 'Genre', 'Type']]

# Filter rows based on the 'Type' column
training_data_3 = selected_features_3[selected_features_3['Type'] == 'Train']
test_data_3 = selected_features_3[selected_features_3['Type'] == 'Test']

# Separate features and labels for both training and test datasets
X_train_3 = training_data_3[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'spectral_rolloff_var']]

X_test_3 = test_data_3[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'spectral_rolloff_var']]

# Convert datasets into numpy
X_train_NP_3 = X_train_3.values

X_test_NP_3 = X_test_3.values

# Compute vector of distances
distances_3 = []
i = 0
while i < len(X_test_NP_3):
    distances_3.append(np.linalg.norm(X_train_NP_3 - X_test_NP_3[i], axis=1))
    i = i+1

# Find kNN
k_nearest_neighbor_id_3 = []
i = 0
while i < len(distances_3):
    k_nearest_neighbor_id_3.append(distances_3[i].argsort()[:k])
    i = i+1

# Connect values with genre
k_nearest_neighbor_genre_3 = []
i = 0
while i < len(k_nearest_neighbor_id_3):
    k_nearest_neighbor_genre_3.append(y_train_NP[k_nearest_neighbor_id_3[i]])
    i = i+1

# Make a prediction of genre
genre_prediction_3 = []
i = 0
while i < len(k_nearest_neighbor_genre_3):
    genre_prediction_3.append(ss.mode(k_nearest_neighbor_genre_3[i])[0])
    i = i+1

# Create confusion matrix
i = 0
confusion_matrix_3 = np.zeros((len(classes), len(classes)))
for i in range(len(classes)):
    for j in range(len(classes)):
      confusion_matrix_3[i, j] = np.sum((y_test_NP == classes[i]) & (np.array(genre_prediction_3).reshape((len(genre_prediction_3))) == classes[j]))

# Compress confusion matrix to find TP, TN, FP and FN for each class
i = 0
j = 0
tp_tn_fp_fn_matrix_3 = np.zeros((len(confusion_matrix_3[0]), 4))
for i in range(len(confusion_matrix_3[0])):
    for j in range(len(confusion_matrix_3[0])):
        if i == j:
            tp_tn_fp_fn_matrix_3[i][0] = confusion_matrix_3[i][j]
        else:
            tp_tn_fp_fn_matrix_3[i][3] = tp_tn_fp_fn_matrix_3[i][3] + confusion_matrix_3[i][j]
            tp_tn_fp_fn_matrix_3[j][2] = tp_tn_fp_fn_matrix_3[i][2] + confusion_matrix_3[i][j]
        
sum_confusion_matrix_3 = sum(list(map(sum, confusion_matrix_3)))
tp_tn_fp_fn_matrix_3[:, 1] = sum_confusion_matrix_3 - (tp_tn_fp_fn_matrix_3[:, 3] + tp_tn_fp_fn_matrix_3[:, 2] + tp_tn_fp_fn_matrix_3[:, 0])

# Find TP, TN, FP and FN for the whole system
i = 0
system_tp_tn_fp_fn_matrix_3 = np.zeros(len(tp_tn_fp_fn_matrix_3[0]))
for i in range(len(tp_tn_fp_fn_matrix_3[0])):
    system_tp_tn_fp_fn_matrix_3[i] = sum(tp_tn_fp_fn_matrix_3[:, i])

# Calculate accuracy and error rate
result_accuracy_3 = system_tp_tn_fp_fn_matrix_3[0]/sum_confusion_matrix_3
result_error_rate_3 = 1 - result_accuracy_3