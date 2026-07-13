# ==========================================
# Financial Fraud Detection using ML
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest, RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Financial_Transaction.csv")

# Remove unnecessary columns
df.drop(columns=["nameOrig","nameDest","IsolationForest_Prediction"],
        errors="ignore",
        inplace=True)

# Encode categorical column
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Features & Target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Scaling
X = StandardScaler().fit_transform(X)

# Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Models
# ==========================================

print("Dataset loaded and preprocessed!")
print(f"Training set size: {X_train.shape[0]:,} samples")
print(f"Test set size: {X_test.shape[0]:,} samples")
print(f"Features: {X_train.shape[1]}")
print("\n" + "="*50)
print("STARTING MODEL TRAINING")
print("="*50 + "\n")

models = {

    "Isolation Forest":
        IsolationForest(contamination=0.01,random_state=42),

    "Random Forest":
        RandomForestClassifier(n_estimators=100,
                               random_state=42)
}

results=[]

predictions={}

# ==========================================
# Train & Evaluate
# ==========================================

for name,model in models.items():

    print(f"Training {name}...", end=" ", flush=True)

    if name=="Random Forest":

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

    else:

        model.fit(X_train)

        pred=model.predict(X_test)

        pred=np.where(pred==-1,1,0)

    predictions[name]=pred

    results.append([

        name,

        accuracy_score(y_test,pred),

        precision_score(y_test,pred),

        recall_score(y_test,pred),

        f1_score(y_test,pred)

    ])
    
    print("✓ Complete")

# ==========================================
# Performance Table
# ==========================================

results=pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ]

)

print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(results.to_string(index=False))
print("="*50 + "\n")

results.to_csv("Model_Performance.csv",index=False)
print("✓ Performance table saved to 'Model_Performance.csv'\n")

# ==========================================
# Comparison Chart
# ==========================================

plot_df=results.melt(

    id_vars="Model",

    var_name="Metric",

    value_name="Score"

)

plt.figure(figsize=(10,6))

sns.barplot(

    data=plot_df,

    x="Model",

    y="Score",

    hue="Metric"

)

plt.title("Machine Learning Model Performance")

plt.ylim(0,1.05)

plt.grid(axis="y",linestyle="--",alpha=.4)

plt.tight_layout()

plt.savefig("Model_Comparison.png", dpi=100, bbox_inches='tight')
print("✓ Model comparison chart saved to 'Model_Comparison.png'")

# ==========================================
# Random Forest Confusion Matrix
# ==========================================

cm=confusion_matrix(

    y_test,

    predictions["Random Forest"]

)

print("\n" + "="*50)
print("CONFUSION MATRIX - RANDOM FOREST")
print("="*50)
print("\n                Predicted Non-Fraud  Predicted Fraud")
print(f"Actual Non-Fraud         {cm[0,0]}           {cm[0,1]}")
print(f"Actual Fraud              {cm[1,0]}           {cm[1,1]}")
print()

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    cbar=False

)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks([0.5,1.5],["Non Fraud","Fraud"])

plt.yticks([0.5,1.5],["Non Fraud","Fraud"],rotation=0)

plt.tight_layout()

plt.savefig("Confusion_Matrix.png", dpi=100, bbox_inches='tight')
print("✓ Confusion matrix saved to 'Confusion_Matrix.png'\n")

print("="*50)
print("CLASSIFICATION REPORT - RANDOM FOREST")
print("="*50)

print(

classification_report(

y_test,

predictions["Random Forest"]

)

)

print("\n✓ Script completed successfully!")