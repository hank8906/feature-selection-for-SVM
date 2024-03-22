import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 加载数据
data_csv_path = r"C:\Users\Hank\DS\dataset\Heart_disease_cleveland_new.csv"
df = pd.read_csv(data_csv_path)

# 分割特征和标签，假设最后一列是标签
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用ANOVA进行特征选择
k_best = SelectKBest(score_func=f_classif, k=6)  # 选择最重要的5个特征
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

# 获取选择的特征的索引
selected_feature_indices = k_best.get_support(indices=True)

# 获取选择的特征名称
selected_feature_names = df.columns[:-1][selected_feature_indices]

print("ANOVA:")
print("Selected Features:")
print(selected_feature_names)

# 建立SVM模型
param_grid = [
  {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

gscv = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 使用选定的特征进行网格搜索
gscv.fit(X_train_selected, y_train)

# print(f"Best hyperparameters: {gscv.best_params_}")
# print(f"Best cross-validation score: {gscv.best_score_:.2f}")

best_model = gscv.best_estimator_

# 使用选定的特征评估测试集
test_accuracy = best_model.score(X_test_selected, y_test)


# 使用选定的特征评估测试集
y_pred = best_model.predict(X_test_selected)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 计算敏感性
sensitivity = recall_score(y_test, y_pred)

# 计算特异性
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# 计算F-measure
f_measure = f1_score(y_test, y_pred)


print(f"Test accuracy with best params: {test_accuracy:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F-measure: {f_measure:.2f}")
