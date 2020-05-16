# NaiveBayes 朴素贝叶斯

import numpy as np
import math
import pickle
from sklearn.metrics import accuracy_score, f1_score

# 加载数据集
from sklearn.datasets import fetch_20newsgroups

# 挑选部分数据集来进行bug测试
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
              'talk.religion.misc', 'sci.electronics']
# 'rec.sport.baseball', 'sci.space', 'talk.politics.guns', 'sci.med']
# 训练集
#newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove='header')
newsgroups_train = fetch_20newsgroups(subset='train', remove='header')
# 测试集
#newsgroups_test = fetch_20newsgroups(subset='test',categories=categories,remove='header')
newsgroups_test = fetch_20newsgroups(subset='test', remove='header')

# 导入本地停用词表
with open("D:\大二下_课程资料\机器学习实验\skl\stopwords.txt", "rb") as f:
    stpwrdlst = f.read()
# 将文本转为TF-IDF向量
from sklearn.feature_extraction.text import TfidfVectorizer

# 停用词为stopwords.txt，全部转换为小写，选择词频为前3000的作为特征，构造稀疏矩阵
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, lowercase=True, max_features=2000)
# 训练集对应稀疏矩阵
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
# 测试集对应稀疏矩阵
vectors_test = vectorizer.transform(newsgroups_test.data)

# 训练集转换后的矩阵大小
print(vectors_train.shape)
# 测试集转换后的矩阵大小
print(vectors_test.shape)
# 非零特征的个数
print(vectors_train.nnz / float(vectors_train.shape[0]))
print(vectors_test.nnz / float(vectors_test.shape[0]))

train_x = vectors_train  # 训练集数据
train_y = newsgroups_train.target  # 训练集标签
test_x = vectors_test  # 测试集数据
test_y = newsgroups_test.target  # 测试集标签


class NaiveBayes:
    def __init__(self, num_class, alpha):
        self.classes = num_class  # 保存分类的种类数
        self.prior = np.zeros(num_class)  # 存储P(Y)
        self.Pxy = np.zeros((num_class, 20000))  # 存储后验概率P(X|Y)
        self.alpha = alpha

    # 求每篇文档属于哪一个类的先验概率P（Y）
    def calculate_prior(self, data):
        for i in range(self.classes):
            sum = 0
            for j in range(data.shape[0]):
                if data[j] == i:
                    sum = sum + 1
            self.prior[i] = sum / data.shape[0]
        return self.prior

    def train(self, x, y):
        print('start training')
        # 首先根据训练集估计P(Y)
        self.prior = self.calculate_prior(y)
        # print(self.prior)
        # 保存每个类的先验概率
        pickle.dump(self.prior, open('NB_prior.pkl', 'wb'))
        # 然后计算在每一个类别Y下每一个属性X的后验概率
        for i in range(self.classes):
            m = []  # 存储属于该类的样本
            sum_m = 0.
            sum_c = 0  # 属于该类的样本
            for j in range(y.shape[0]):
                if y[j] == i:
                    sum_c = sum_c + 1
                    # 从稀疏矩阵中提取对应的行向量
                    temp = np.zeros(x.shape[1])
                    for k in range(x.indptr[j], x.indptr[j + 1]):
                        temp[x.indices[k]] = x.data[k]
                    m.append(temp)
                    sum_m += np.sum(temp)
            print('属于类别' + str(i) + '的有' + str(sum_c) + '个')
            m = np.array(m).T  #转置
            for j in range(x.shape[1]):  # 遍历所有特征
                sum = np.sum(m[j])
                # 贝叶斯估计求出条件概率pxy,加入拉普拉斯平滑
                self.Pxy[i][j] = (100000.0) * (sum + self.alpha) / (sum_m + x.shape[1] * self.alpha)

    def predict(self, x):
        print('start predicting')
        pred_labels = [None for _ in range(x.shape[0])]
        for i in range(x.shape[0]):
            temp = np.zeros(x.shape[1])
            for j in range(x.indptr[i], x.indptr[i + 1]):
                temp[x.indices[j]] = x.data[j]
            Pyx = np.zeros(self.classes)  # 存储后验概率P(Y|X)
            for j in range(self.classes):
                Pyx[j] = np.log(self.prior[j])
                for k in range(self.Pxy.shape[1]):
                    if self.Pxy[j][k] != 0:
                        Pyx[j] += np.log(self.Pxy[j][k] * temp[k] + 1)
            # 概率最大的作为预测分类
            pred_labels[i] = np.argmax(Pyx)
        return pred_labels


from sklearn.metrics import classification_report
if __name__ == "__main__":
    classes = np.unique(train_y)  # 去掉多余项，得到分类的种类数
    class_nums = len(classes)
    model = NaiveBayes(num_class=class_nums,alpha=1)
    # 训练模型
    model.train(train_x, train_y)

    # 存储模型
    pickle.dump(model, open('NB_models.pkl', 'wb'))
    # 计算准确率
    pred = model.predict(test_x)
    # 计算准确率
    sum = 0
    for i in range(test_x.shape[0]):
        if pred[i] == test_y[i]:
            sum = sum + 1
    accuracy = sum / float(test_x.shape[0])
    print('NaiveBayes_accuracy = ', accuracy)
    print(classification_report(test_y, pred, target_names=newsgroups_test.target_names))
