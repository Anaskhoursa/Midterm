import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np






#>>>>>>>>> Data Preprocessing 


# load the dataset
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# strip extra spaces in column names
df.columns = df.columns.str.strip()



# show basic information
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns.tolist())
print("\nFirst 5 Rows:\n", df.head())

# check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

# drop rows with missing values (if any)
df = df.dropna()
print("\nNew Dataset Shape after removing missing values:", df.shape)

# replace inf/-inf with NaN and drop them
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

selected_features = [
    'Flow Duration', 
    'Total Fwd Packets', 
    'Total Backward Packets',
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Label'  # Target variable
]

# create a new dataframe with only selected features
df_selected = df[selected_features]
print("\n Selected Features DataFrame:\n", df_selected.head())


# encode the "Label" column
le = LabelEncoder()
df_selected['Label'] = le.fit_transform(df_selected['Label'])

# map encoded labels back to readable form (0 = BENIGN, 1 = ATTACK)
print("\nLabel Mapping:")
for index, class_ in enumerate(le.classes_):
    print(f"{index} -> {class_}")




# >>>>>>> Exploratory Data Analysis

print("\nSummary Statistics:\n", df_selected.describe())

# count of attack types
attack_counts = df['Label'].value_counts()
print("\nAttack Type Distribution:\n", attack_counts)

# visualize distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Label', order=df['Label'].value_counts().index, palette='Set2')
plt.title("Attack Type Distribution")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

features_to_plot = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']

for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=50, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# compute correlation matrix for selected numeric features
corr_matrix = df_selected.drop('Label', axis=1).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# >>>>>> Data Mining (Classification)


# split features and labels
X = df_selected.drop("Label", axis=1)
y = df_selected["Label"]

# split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# initialize and train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# predict on test data
y_pred = model.predict(X_test)


# print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()



