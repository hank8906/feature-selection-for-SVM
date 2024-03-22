import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from skrebate import ReliefF

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

# 使用ReliefF进行特征选择
reliefF = ReliefF()
X_train_selected = reliefF.fit_transform(X_train_scaled, y_train)
X_test_selected = reliefF.transform(X_test_scaled)

# 获取选择的特征索引
selected_features_indices = reliefF.top_features_

# 获取选择的特征名称
selected_feature_names = df.columns[:-1][selected_features_indices]

# 选择特定数量的特征
num_selected_features = 6  # 设置要选择的特征数量
selected_features_indices = selected_features_indices[:num_selected_features]
selected_feature_names = selected_feature_names[:num_selected_features]

print("ReliefF")
print("Selected Features:")
print(selected_feature_names)

# 更新训练和测试集以仅包含选定的特征
X_train_selected = X_train_scaled[:, selected_features_indices]
X_test_selected = X_test_scaled[:, selected_features_indices]

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
print(f"Test accuracy with best params: {test_accuracy:.2f}")

# 使用选定的特征进行预测
y_pred = best_model.predict(X_test_selected)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 计算敏感性
sensitivity = recall_score(y_test, y_pred)

# 计算特异性
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# 计算F-measure
f_measure = f1_score(y_test, y_pred)

print(f"Specificity: {specificity:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"F-measure: {f_measure:.2f}")
