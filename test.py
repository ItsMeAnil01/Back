import pandas as pd
data = pd.read_csv('D:/IpBack/heart.csv')
print(data.shape)  # Should print (305, 16)
print(data.columns)  # Should include 'target', 'area', 'pri', and EXPECTED_FEATURES
print(data.isnull().sum())  # Should show no missing values