# Cao Hoàng Khôi Nguyên - ML03

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("C:/Users/caong/OneDrive - Thanh Thiếu Niên Đại Đạo/KN/Cybersoft Academy/Machine Learning/Thực hành/14-01-2024-11-43-57-tuan_01_thuchanh_demo_baitap/house_price.csv")

# Question 1
population_std = np.std(df['SalePrice'])  # Standard deviation of the population
num_samples = 400  # Number of samples
sample_size = 100  

# Calculate sample means
sample_means = [np.mean(df['SalePrice'].sample(sample_size)) for _ in range(num_samples)] # Mean of SalePrice
#print (sample_means)

# Calculate standard deviation of sample means
sample_means_std = np.std(sample_means)

# Draw
#plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='lightblue', edgecolor='black', linewidth=1.2)
#plt.title('Distribution of Sample Means')
#plt.xlabel('Sample Mean')
#plt.ylabel('Density')

#plt.show() #Show

#print("Standard deviation of sample means using CLT:", sample_means_std) # Standard Deviation of SalePrice


# Question 2

# SalePrice and LotShape
#plt.figure(figsize=(10, 6))
#sns.boxplot(x='LotShape', y='SalePrice', data=df)
#plt.title('SalePrice distribution by LotShape')
#plt.xlabel('LotShape')
#plt.ylabel('SalePrice')
#plt.show()

# SalePrice and Street
#plt.figure(figsize=(10, 6))
#sns.boxplot(x='Street', y='SalePrice', data=df)
#plt.title('SalePrice distribution by Street')
#plt.xlabel('Street')
#plt.ylabel('SalePrice')
#plt.show()

# SalePrice and LotConfig
#plt.figure(figsize=(10, 6))
#sns.boxplot(x='LotConfig', y='SalePrice', data=df)
#plt.title('SalePrice distribution by LotConfig')
#plt.xlabel('LotConfig')
#plt.ylabel('SalePrice')
#plt.show()


# Question 2
# SalePrice and LotArea
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x='LotArea', y='SalePrice', data=df)
#plt.title('SalePrice distribution by LotArea')
#plt.xlabel('LotArea')
#plt.ylabel('SalePrice')
#plt.show()

# SalePrice and LotFrontage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LotFrontage', y='SalePrice', data=df)
plt.title('SalePrice distribution by LotFrontage')
plt.xlabel('LotFrontage')
#plt.ylabel('SalePrice')
#plt.show()

# SalePrice and MasVnrArea
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=df)
plt.title('SalePrice distribution by MasVnrArea')
plt.xlabel('MasVnrArea')
plt.ylabel('SalePrice')
plt.show()


# Question 3
# SalePrice and Street
onehot_encoder = OneHotEncoder(sparse=False)
street_encoded = onehot_encoder.fit_transform(df[['Street']])
# Convert the encoded array into a DataFrame
street_encoded_df = pd.DataFrame(street_encoded, columns=onehot_encoder.get_feature_names_out(['Street']))
# Concatenate the encoded DataFrame with the original DataFrame
df_encoded = pd.concat([df.drop(columns=['Street']), street_encoded_df], axis=1)

# Question 4
df['LotArea'].fillna(df['LotArea'].mean(), inplace=True)
df['SalePrice'].fillna(df['SalePrice'].mean(), inplace=True)
df = df.drop_duplicates()
# Draw scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LotArea', y='SalePrice', data=df)
plt.title('SalePrice distribution by LotArea')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.show()