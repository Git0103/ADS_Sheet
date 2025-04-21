import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)

# Load data
df = pd.read_csv('/content/placement.csv')

# Split data
X = df[['cgpa', 'placement_exam_marks']]
y = df['placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:,1]

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# CGPA distribution
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='cgpa', bins=20, kde=True, hue='placed', multiple='stack')
plt.title('CGPA Distribution by Placement Status')
plt.xlabel('CGPA')
plt.ylabel('Count')

# Placement exam marks distribution
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='placement_exam_marks', bins=20, kde=True, hue='placed', multiple='stack')
plt.title('Exam Marks Distribution by Placement Status')
plt.xlabel('Placement Exam Marks')

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='cgpa', y='placement_exam_marks', hue='placed', 
                style='placed', palette={0:'red', 1:'green'}, alpha=0.7)
plt.title('CGPA vs Exam Marks by Placement Status')
plt.legend(title='Placement', labels=['Not Placed', 'Placed'])
plt.show()



import plotly.express as px

# Interactive 3D Scatter Plot
fig = px.scatter_3d(df, x='cgpa', y='placement_exam_marks', z='placed',
                    color='placed', color_continuous_scale=['red', 'green'],
                    title='3D View of Placement Data')
fig.update_layout(scene=dict(zaxis=dict(tickvals=[0, 1], ticktext=['Not Placed', 'Placed'])))
fig.show()