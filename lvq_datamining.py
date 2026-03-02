import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

dataset = pd.read_csv('ispu_dki1.csv')
dataset

data_raw = dataset[['pm10', 'so2', 'co', 'o3', 'no2', 'categori']].copy()
print("\nUkuran dataset:", data_raw.shape)

data_raw = data_raw[data_raw['categori'] != 'TIDAK ADA DATA']
print("\nUkuran dataset setelah hapus 'TIDAK ADA DATA':", data_raw.shape)

print("\nMissing value SEBELUM imputasi:")
print(data_raw.isna().sum())

# Imputasi NaN dengan nilai rata-rata (mean)
data_raw.fillna(data_raw.mean(numeric_only=True), inplace=True)
print("\nMissing value SETELAH imputasi:")
print(data_raw.isna().sum())

# Konversi kolom numerik
for col in ['pm10', 'so2', 'co', 'o3', 'no2']:
    data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce')

# Hapus outlier berdasarkan IQR
def remove_outliers_iqr_full(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df

data_cleaned = remove_outliers_iqr_full(data_raw, ['pm10', 'so2', 'co', 'o3', 'no2'])

print("Ukuran dataset setelah hapus outlier:", data_cleaned.shape)

for column in ['pm10', 'so2', 'co', 'o3', 'no2']:
    print(f"\n--- {column.upper()} ---")
    print("Distribusi SEBELUM outlier dihapus:")
    print(data_raw[column].value_counts().sort_index())
    print("\nDistribusi SETELAH outlier dihapus:")
    print(data_cleaned[column].value_counts().sort_index())
    print('\n' + '='*60 + '\n')

x = data_cleaned[['pm10', 'so2', 'co', 'o3', 'no2']]
y = data_cleaned['categori']

# Encode label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping label:", label_mapping)

# Min-Max
mms = MinMaxScaler()
x_scaled = mms.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

# Visualisasi histogram setelah normalisasi
plt.figure(figsize=(20, 20))
x_scaled.hist(bins=30, grid=False, figsize=(20, 20))
plt.suptitle('Histogram Setelah Normalisasi Min-Max', fontsize=16)
plt.show()

class LVQ(object):
    def __init__(self, sizeInput, sizeOutput, max_epoch, alpha=0.3):
        self.sizeInput = sizeInput
        self.sizeOutput = sizeOutput
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.weight = np.zeros((sizeOutput, sizeInput))

    def getWeight(self):
        return self.weight

    def getAlpha(self):
        return self.alpha

    def train(self, train_data, train_target):
        weight_label, label_index = np.unique(train_target, return_index=True)
        self.weight = train_data[label_index].astype(np.float64)
        train_data = np.delete(train_data, label_index, axis=0)
        train_target = np.delete(train_target, label_index, axis=0)

        for epoch in range(1, self.max_epoch + 1):
            for data, target in zip(train_data, train_target):
                distance = np.sqrt(np.sum((data - self.weight) ** 2, axis=1))
                idx_min = np.argmin(distance)
                if target == weight_label[idx_min]:
                    self.weight[idx_min] += self.alpha * (data - self.weight[idx_min])
                else:
                    self.weight[idx_min] -= self.alpha * (data - self.weight[idx_min])
            self.alpha *= (1 - epoch / self.max_epoch)

        return (self.weight, weight_label)

    def test(self, test_data, weight_class):
        weight, label = weight_class
        output = []
        for data in test_data:
            distance = np.sqrt(np.sum((data - weight) ** 2, axis=1))
            idx_min = np.argmin(distance)
            output.append(label[idx_min])
        return output
    
# Pembagian data training & testing
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=100)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print("Distribusi sebelum undersampling:", Counter(y_train))
undersample = RandomUnderSampler(sampling_strategy={0: 716, 1: 1000, 2: 61}, random_state=42)
x_train_resampled, y_train_resampled = undersample.fit_resample(x_train, y_train)
print("Distribusi setelah undersampling:", Counter(y_train_resampled))

n_input = x_train.shape[1]
n_output = len(np.unique(y_train))
print('Input Neuron:', n_input)
print('Output Neuron:', n_output)

# Training
n_input = x_train_resampled.shape[1]
n_output = len(np.unique(y_train_resampled))

lvq = LVQ(sizeInput=n_input, sizeOutput=n_output, max_epoch=20)
x_train_resampled = np.array(x_train_resampled)
y_train_resampled = np.array(y_train_resampled)

bobot_dan_label = lvq.train(x_train_resampled, y_train_resampled)
bobot = lvq.getWeight()
print("Bobot akhir:", bobot)

# Testing
y_pred = lvq.test(x_test, bobot_dan_label)
print('Accuracy:', accuracy_score(y_test, y_pred))

y_test_encoded = y_test
y_pred_encoded = y_pred

# Confusion matrix
cf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
labels = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix LVQ")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(20, 20))
sns.boxplot(data=x_scaled, orient='h')
plt.title('Boxplot', fontsize=16)
plt.xlabel('Nilai')
plt.ylabel('Fitur')
plt.show()

plt.figure(figsize=(20, 20))
x_scaled.hist(bins=30, grid=False, figsize=(20, 20), edgecolor='black', color='skyblue')
plt.suptitle('Histogram', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Uji korelasi atribut
f, ax = plt.subplots(figsize=(20, 20))
corr = x_scaled.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f', linewidths=.05)