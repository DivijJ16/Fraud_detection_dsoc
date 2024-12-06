import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Prepare Data
# file_path = 'creditcard.csv'  # Replace with your file path
file_path = 'creditcard.csv'  
data = pd.read_csv(file_path)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Separate features (V1 to Amount) and target (Class)
X = data.drop(columns=['Class', 'Time'])  # Drop 'Time' as it's unlikely to be useful
y = data['Class']

#Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Apply SMOTE to balance the training data
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_train_resampled))

# Step 3: Train the Logistic Regression Model

model = LogisticRegression(max_iter=1000)
# model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# model = LGBMClassifier(random_state=42, class_weight='balanced')
# model = SVC(kernel='rbf', probability=True, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# AUC-ROC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {roc_auc:.4f}")
