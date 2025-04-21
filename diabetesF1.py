import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model BEFORE SMOTE
model_before = RandomForestClassifier(random_state=42)
model_before.fit(X_train, y_train)
y_pred_before = model_before.predict(X_test)
f1_before = f1_score(y_test, y_pred_before)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model AFTER SMOTE
model_after = RandomForestClassifier(random_state=42)
model_after.fit(X_resampled, y_resampled)
y_pred_after = model_after.predict(X_test)
f1_after = f1_score(y_test, y_pred_after)

# Print F1-scores
print("F1-score BEFORE SMOTE:", f1_before)
print("F1-score AFTER SMOTE:", f1_after)
