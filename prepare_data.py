import pandas as pd

# Load the original datasets
df_train = pd.read_csv('data/UNSW_NB15_training-set.csv')
df_test = pd.read_csv('data/UNSW_NB15_testing-set.csv')

# Filter for 'DoS' and 'Normal' attack categories
df_train_filtered = df_train[df_train['attack_cat'].isin(['DoS', 'Normal'])]
df_test_filtered = df_test[df_test['attack_cat'].isin(['DoS', 'Normal'])]

# Save the filtered data to new CSV files
df_train_filtered.to_csv('data/UNSW_NB15_DoS_train_data.csv', index=False)
df_test_filtered.to_csv('data/UNSW_NB15_DoS_test_data.csv', index=False)

print("Filtered data saved successfully.")
