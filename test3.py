import pandas as pd

# Read both datasets
data_10s = pd.read_csv('Classification music/GenreClassData_10s.txt', sep='\t')
data_30s = pd.read_csv('Classification music/GenreClassData_30s.txt', sep='\t')

# Concatenate the datasets
combined_data = pd.concat([data_10s, data_30s], ignore_index=True)

# Display the first 5 rows of the combined_data
print(combined_data.head())

# Display the last 5 rows of the combined_data
print(combined_data.tail())

# Display a summary of the combined_data (number of non-null values, data types, memory usage)
print(combined_data.info())

# Display the shape of the combined_data (number of rows and columns)
print(combined_data.shape)

# Display summary statistics for the combined_data (mean, standard deviation, min, max, etc.)
print(combined_data.describe())