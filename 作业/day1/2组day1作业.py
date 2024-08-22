import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 从指定路径读取数据
data_path = 'C:\\Users\\16708\\Downloads\\train.csv'
data = pd.read_csv(data_path)

# 创建数据的副本
df = data.copy()

# 随机显示10行数据
print(df.sample(10))

# 去除无用特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# 查看数据的信息
print(df.info())

# 检查是否有缺失值
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))

# 删除缺失值
df.dropna(inplace=True)

# 再次检查缺失值
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))

# 将分类数据转化为数值型数据
df = pd.get_dummies(df)

# 分离特征和目标变量
X = df.drop(columns=['Survived'])
y = df['Survived']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化并训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(conf_matrix)

# 生成分类报告
class_report = classification_report(y_test, y_pred)
print("分类报告:")
print(class_report)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # 保存为图片
plt.show()

# 预测单个乘客的生存概率示例
# 假设你有一个新的乘客数据
# 这里的列顺序必须和训练数据中的特征列顺序一致
new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex_female': [1],
    'Sex_male': [0],
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked_C': [0],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})

# 确保新乘客数据的列顺序与训练数据一致
new_passenger = new_passenger[X.columns]

# 标准化新乘客数据
new_passenger = scaler.transform(new_passenger)

# 预测生存概率
probability = model.predict_proba(new_passenger)[0][1]
print(f'某乘客的生存概率: {probability:.2f}')
