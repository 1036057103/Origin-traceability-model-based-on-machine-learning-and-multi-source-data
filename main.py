import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import openpyxl
import matplotlib

matplotlib.use('TkAgg')

# Set font
plt.rcParams['font.family'] = 'Times New Roman'

# Load data
file_path = 'data.xlsx'
data = pd.read_excel(file_path)

# Extract feature variables and target variables
X = data.iloc[:, 2:].values  # feature variables
y = data.iloc[:, 1].values  # target variables

# Custom labels
custom_labels = ['AC', 'AL_HB_Wild', 'AL_HN_Wild', 'AL_HN_Cultivated', 'AL_JS_Cultivated', 'AL_HB_Cultivated',
                 'AL_JS_Wild']

# 1. Analyze data distribution
print("=== Analyze data distribution ===")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"category {custom_labels[u]}: {c} samples ({c / len(y) * 100:.2f}%)")

# Calculate category weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
print(f"\n Category weight: {class_weight_dict}")

# Stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=72)

# 2. Use feature selection more suitable for imbalanced data
lgb_model = lgb.LGBMClassifier(class_weight='balanced')
rfecv = RFECV(
    estimator=lgb_model,
    step=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro'
)
rfecv.fit(X_train, y_train)
print(f"Optimal number of features: {rfecv.n_features_}")

# Filter features
X_train_rfe = rfecv.transform(X_train)
X_test_rfe = rfecv.transform(X_test)

# Visualizing the feature selection process
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (macro-average F1)")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("feature_selection_process.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Define the model - add weights to models that support category weights
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(class_weight='balanced', random_state=42),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(class_weight='balanced', random_state=42)
}

# 4. Use evaluation metrics more suitable for imbalanced data
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy']

# Initialize result dictionary
results = {model_name: {score: [] for score in scoring} for model_name in models.keys()}
detailed_results = []

# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=82)

print("\n=== Start model training and evaluation ===")
for model_name, model in models.items():
    print(f"\n Training model: {model_name}")

    try:
        model.fit(X_train_rfe, y_train)

        for score_name in scoring:
            scores = cross_val_score(model, X_train_rfe, y_train, cv=kf, scoring=score_name)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results[model_name][score_name].append(mean_score)
            results[model_name][score_name].append(std_score)

        y_pred = model.predict(X_test_rfe)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        test_recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        test_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Store detailed results
        detailed_results.append({
            'Model': model_name,
            'Test Accuracy': test_accuracy,
            'Test Precision (Macro)': test_precision_macro,
            'Test Recall (Macro)': test_recall_macro,
            'Test F1 (Macro)': test_f1_macro
        })

        print(f"Test set F1: {test_f1_macro:.4f}")

    except Exception as e:
        print(f"model {model_name} training failed: {e}")
        continue

# 5. Save results to Excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Model evaluation results"

header = ["model"]
for score in scoring:
    header.extend([f"{score} (Mean)", f"{score} (SD)"])
ws.append(header)

for model_name in models.keys():
    if model_name in results:
        row = [model_name]
        for score in scoring:
            if results[model_name][score]:
                row.extend([f"{results[model_name][score][0]:.4f}", f"{results[model_name][score][1]:.4f}"])
            else:
                row.extend(["N/A", "N/A"])
        ws.append(row)

ws2 = wb.create_sheet("Detailed test set results")
ws2.append(["model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"])
for result in detailed_results:
    ws2.append([
        result['Model'],
        f"{result['Test Accuracy']:.4f}",
        f"{result['Test Precision (Macro)']:.4f}",
        f"{result['Test Recall (Macro)']:.4f}",
        f"{result['Test F1 (Macro)']:.4f}"
    ])

wb.save("improved_results.xlsx")
print("\n Model evaluation results have been saved to 'improved_results.xlsx'")

# 6. Generate confusion matrix and classification reports
print("\n=== Generate confusion matrix and classification reports ===")
for model_name, model in models.items():
    if model_name not in [r['Model'] for r in detailed_results]:
        continue

    print(f"\n{'=' * 50}")
    print(f"model: {model_name}")
    print(f"{'=' * 50}")

    try:
        model.fit(X_train_rfe, y_train)
        y_pred = model.predict(X_test_rfe)

        # Generate classification report
        print("Detailed classification report:")
        print(classification_report(y_test, y_pred, target_names=custom_labels, zero_division=0))

        # Calculate accuracy for each category
        cm = confusion_matrix(y_test, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("Accuracy for each category:")
        for i, (label, acc) in enumerate(zip(custom_labels, class_accuracy)):
            print(f"  {label}: {acc:.4f}")

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_labels)
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
        plt.title(f"{model_name}", fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"improved_confusion_matrix_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Error generating visualization for {model_name}: {e}")
        continue

print("\n=== 分析完成 ===")
print("生成的文件:")
print("- improved_results.xlsx: 详细的模型评估结果")
print("- feature_selection_process.png: 特征选择过程图")
print("- improved_confusion_matrix_*.png: 各模型的混淆矩阵")
