import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, \
    precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import re
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 自定义停用词列表
custom_stop_words = set([
    "the", "and", "is", "in", "to", "of", "that", "it", "you", "he", "was", "for", "on", "are", "as", "with", "his",
    "they",
    "be", "at", "one", "have", "this", "from", "or", "had", "by", "not", "word", "but", "what", "some", "we", "can",
    "out",
    "other", "were", "which", "do", "their", "time", "if", "will", "how", "said", "an", "each", "when", "up", "use",
    "your",
    "would", "see", "more", "could", "has", "been", "than", "most", "should", "about", "so", "can", "may", "like",
    "very"
])

# 打印当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 使用绝对路径读取数据
train_data_path = r'D:\\PYCHARM\\垃圾信息识别\\data\\train.txt'
test_data_path = r'D:\\PYCHARM\\垃圾信息识别\\data\\test.txt'
test_labels_path = r'D:\\PYCHARM\\垃圾信息识别\\data\\test_label.csv'

print(f"读取训练数据从: {train_data_path}")
print(f"读取测试数据从: {test_data_path}")
print(f"读取测试标签从: {test_labels_path}")

try:
    # 假设 train.txt 和 test.txt 的格式是两列：id 和 text
    # 使用 pandas 的 read_csv 函数加载训练数据，指定分隔符为制表符(\t)，无表头，自定义列名为 ['label', 'text']
    train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['label', 'text'])
    test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['id', 'text'])
    # 加载测试标签，假设没有表头，自定义列名为 ['label']
    test_labels = pd.read_csv(test_labels_path, header=None, names=['label'])
    print(f"成功加载 {train_data_path}。形状: {train_data.shape}")
    print(f"成功加载 {test_data_path}。形状: {test_data.shape}")
    print(f"成功加载 {test_labels_path}。形状: {test_labels.shape}")
except FileNotFoundError as e:
    # 如果发生 FileNotFoundError 异常，表示文件未找到，则打印错误信息并退出程序
    print(f"文件未找到: {e}")
    exit(1)

# 数据对齐
# 检查测试数据和测试标签的数量是否匹配，如果测试数据多于标签则截断测试数据
if len(test_data) > len(test_labels):
    print(f"截断 test_data 从 {len(test_data)} 行到 {len(test_labels)} 行。")
    test_data = test_data.iloc[:len(test_labels)]
    test_data = test_data.iloc[:len(test_labels)]
# 如果测试标签多于测试数据，则截断测试标签
elif len(test_data) < len(test_labels):
    print(f"截断 test_labels 从 {len(test_labels)} 行到 {len(test_data)} 行。")
    test_labels = test_labels.iloc[:len(test_data)]

# 合并 test_data 和 test_labels
test_data['label'] = test_labels['label']

# 确保 test_data 中的 'text' 是字符串
test_data['text'] = test_data['text'].astype(str)

# 标签编码
# 创建 LabelEncoder 实例，用于将分类标签转换为数值形式
label_encoder = LabelEncoder()
# 对训练数据中的 'label' 列进行标签编码，并更新该列
train_data['label'] = label_encoder.fit_transform(train_data['label'])
# 对测试数据中的 'label' 列进行标签编码，注意这里使用的是 transform 而不是 fit_transform，
# 因为我们希望保持训练集和测试集标签编码的一致性
test_data['label'] = label_encoder.transform(test_data['label'])
# 从 train_data 和 test_data 中提取标签列作为目标变量 y_train 和 y_test
y_train = train_data['label']
y_test = test_data['label']


# 数据预处理和增强
def preprocess_text(text):
    # 去除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 去除数字
    text = re.sub(r'\d+', '', text)
    # 分词
    words = text.split()
    # 去除常见停用词
    filtered_words = [word for word in words if word not in custom_stop_words]
    return ' '.join(filtered_words)

# 对训练数据集中的 'text' 列的每一个元素应用 preprocess_text 函数，
# 该函数会对每个文本字符串进行预处理，如转换为小写、移除标点符号等。
# 这样可以确保所有文本数据在进入模型之前已经过标准化处理。
train_data['text'] = train_data['text'].apply(preprocess_text)
# 同样的操作应用于测试数据集。对 test_data 中的 'text' 列的每一个元素也应用 preprocess_text 函数，
# 确保测试集中的文本数据经过了与训练集相同的预处理步骤，
# 以保持两者的一致性和可比性。
test_data['text'] = test_data['text'].apply(preprocess_text)

# 文本预处理：使用 n-grams 和停用词过滤
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))#仅保留信息量最大的前5000个特征（词汇），使用单个词 (unigrams) 和两个词的组合 (bigrams) 作为特征
# 对训练集中的 'text' 列应用 fit_transform 方法。
# fit_transform 方法会先根据训练文本数据拟合 (fit) 向量化器，
# 即学习词汇表和计算词频逆文档频率 (TF-IDF) 的权重，
# 然后将文本数据转换 (transform) 成 TF-IDF 特征矩阵。
# X_train_tfidf 将是稀疏矩阵，表示训练集的文本特征。
X_train_tfidf = vectorizer.fit_transform(train_data['text'])       #不仅考虑单独的词，还考虑连续的两个词作为一个整体特征，这有助于捕捉短语和固定表达
# 对测试集中的 'text' 列应用 transform 方法。也是稀疏矩阵，表示训练集的文本特征。
X_test_tfidf = vectorizer.transform(test_data['text'])

# 数据平衡，SMOTE 是一种过采样技术，它通过合成新的少数类样本来进行过采样
smote = SMOTE(random_state=42)# 初始化 SMOTE 实例，设置随机种子为 42 以确保结果可复现。
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)# 使用 fit_resample 方法对训练集进行重采样。并返回重采样后的特征矩阵和目标变量。

# 拆分验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# 定义模型
model_nb = MultinomialNB()# 初始化朴素贝叶斯分类器
model_svc = SVC(probability=True, kernel='linear', random_state=42, class_weight='balanced')# 初始化支持向量机分类器，class_weight='balanced' 自动调整类权重以处理类别不平衡问题
# 初始化随机森林分类器
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')#使用 100 棵决策树进行集成，限制每棵树的最大深度，防止过拟合
# 初始化逻辑回归分类器
model_lr = LogisticRegression(max_iter=1000, C=1.0, penalty='l2', class_weight='balanced')#设置最大迭代次数，确保算法有足够的时间收敛，使用 L2 正则化
# 初始化 K 近邻分类器
model_knn = KNeighborsClassifier(n_neighbors=5)#考虑最近的 5 个邻居来做出分类决策。
# 初始化梯度提升分类器
model_gb = GradientBoostingClassifier(random_state=42)
# 初始化 XGBoost 分类器
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)#禁用内置的标签编码器，指定评估指标为对数损失，指定评估指标为对数损失
# 初始化 LightGBM 分类器
model_lgb = LGBMClassifier(random_state=42)

# 超参数调优 - SVM，使用了 GridSearchCV 方法来进行网格搜索
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_svc = GridSearchCV(SVC(probability=True), param_grid_svc, cv=3, scoring='accuracy')#设置交叉验证的折数为 3 折，使用准确率 ('accuracy') 作为评估指标
# 使用 fit 方法在训练数据上执行网格搜索，通过交叉验证评估每个组合的性能
grid_svc.fit(X_train, y_train)
# 获取并打印最佳模型的参数组合
best_svc = grid_svc.best_estimator_
print(f"SVM 最佳参数: {grid_svc.best_params_}")

# 训练模型
model_nb.fit(X_train, y_train)
best_svc.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
model_knn.fit(X_train, y_train)
model_gb.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_lgb.fit(X_train, y_train)


# 评估模型
def evaluate_model(model, X_val, y_val, model_name):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_prob)

    print(f"{model_name}:")
    print(f'准确率: {acc}')
    print(report)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

    # 绘制 ROC 曲线
    fpr, tpr, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(tpr, fpr)  # 注意这里交换了 fpr 和 tpr 的顺序
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (area = {pr_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall 曲线')
    plt.legend(loc="lower left")
    plt.show()

    return acc


models = [
    ('朴素贝叶斯', model_nb),
    ('支持向量机', best_svc),
    ('随机森林', model_rf),
    ('逻辑回归', model_lr),
    ('K-Nearest Neighbors', model_knn),
    ('Gradient Boosting Classifier', model_gb),
    ('XGBoost Classifier', model_xgb),
    ('LightGBM Classifier', model_lgb)
]

accuracies = []

for name, model in models:
    acc = evaluate_model(model, X_val, y_val, name)# 调用 evaluate_model 函数来评估当前模型在验证集上的表现
    accuracies.append((name, acc))# 将模型名称和其对应的准确率作为一个元组添加到 accuracies 列表中

# 绘制准确率条形图
plt.figure(figsize=(12, 8))
sns.barplot(x=[acc[1] for acc in accuracies], y=[acc[0] for acc in accuracies])
plt.title('模型准确率比较')
plt.xlabel('准确率')
plt.ylabel('模型')
plt.xlim(0, 1)
plt.show()



