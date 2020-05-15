#NaiveBayes朴素贝叶斯

import numpy as np
import pickle
from sklearn.metrics import accuracy_score,f1_score
import math
from scipy.stats import norm
#加载数据集
from sklearn.datasets import fetch_20newsgroups
#挑选部分数据集来进行bug测试
#categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc']
              #'talk.religion.misc','sci.electronics', 'soc.religion.christian',
              #'rec.sport.baseball', 'sci.space','talk.politics.guns', 'sci.med']
#训练集
#newsgroups_train = fetch_20newsgroups(subset='train',categories=categories,remove='header')
newsgroups_train = fetch_20newsgroups(subset='train',remove='header')
#测试集
#newsgroups_test = fetch_20newsgroups(subset='test',categories=categories,remove='header')
newsgroups_test = fetch_20newsgroups(subset='test',remove='header')


#导入本地停用词表
with open("D:\大二下_课程资料\机器学习实验\skl\stopwords.txt", "rb") as f:
    stpwrdlst = f.read()
#将文本转为TF-IDF向量
from sklearn.feature_extraction.text import TfidfVectorizer
# 停用词为stopwords.txt，全部转换为小写，选择词频为前5000的作为特征，构造稀疏矩阵
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,lowercase=True,max_features=2000)
#训练集对应稀疏矩阵
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
#测试集对应稀疏矩阵
vectors_test = vectorizer.transform(newsgroups_test.data)

# 训练集转换后的矩阵大小
print(vectors_train.shape)
# 测试集转换后的矩阵大小
print(vectors_test.shape)
# 非零特征的个数
print(vectors_train.nnz / float(vectors_train.shape[0]))
print(vectors_test.nnz / float(vectors_test.shape[0]))

train_x = vectors_train #训练集数据
train_y = newsgroups_train.target # 训练集标签
test_x = vectors_test # 测试集数据
test_y = newsgroups_test.target #测试集标签


class NaiveBayes:
    def __init__(self,class_num = 20):

        self.prior = np.zeros(class_num) # 存储先验概率
        self.avgs = np.zeros((class_num, 5000))  # 存储训练集均值
        self.vars = np.zeros((class_num, 5000)) # 存储方差
        self.n_class = class_num # 存储类别数量

    # 高斯概率密度函数
    def Gaussian_probability(self,x,mean,stdev):
        return norm.pdf(x, loc = mean, scale = stdev)

    #求每篇文档属于哪一个类的先验概率P（Y）
    def calculate_prior(self, data):
        for i in range(self.n_class):
            sum = 0
            for j in range(data.shape[0]):
                if data[j] == i:
                    sum = sum + 1
            self.prior[i] = sum / data.shape[0]
        return self.prior
    #self.prior = calculate_prior(self,train_y)


    def train(self, data, target):
        #print(self.vars.shape[0])
        #print(self.avgs.shape[0])
        row = data.shape[0]
        col = data.shape[1]
        # 计算先验概率
        self.prior = self.calculate_prior(target)
        # 计算每一个类下的训练集均值与方差
        for i in range(self.n_class):
            m = [] # 存放当前样本
            j = 0
            print('类别 ', i)
            sum = 0
            for j in range(row): # max_j = 1655
                if target[j] == i: #属于当前类别
                    sum = sum + 1
                    temp = np.zeros(col)
                    for k in range(data.indptr[i],data.indptr[i+1]):
                        temp[data.indices[k]] = data.data[k]
                    #print(temp.shape)
                    m.append(temp)
                    #print(len(m))
            print('sum = ', sum)
            m = np.array(m).T
            #print('len = ', len(m))
            #print(np.mean(m[j]))
            #exit()
            for j in range(col): #max_j = 2000
                self.avgs[i][j] = np.mean(m[j])
                self.vars[i][j] = np.var(m[j])

    def predict(self, data):
        eps = 1e-10
        print('Start predicting')
        pred_labels = {} #存在测试集中每个实例的分类
        row = data.shape[0]
        col = data.shape[1]
        for i in range(row):
            m = np.zeros(col)
            bound_0 = data.indptr[i]
            bound_1 = data.indptr[i + 1]
            for j in range(bound_0, bound_1):
                m[data.indices[j]] = data.data[j]
            PYX = np.zeros(self.n_class)
            for j in range(self.n_class):
                PYX[j] = np.log(self.prior[j])
                for k in range(self.vars.shape[1]):
                    miu = self.avgs[j][k]
                    sigma = self.vars[j][k]
                    if miu*sigma == 0:
                        continue
                    PYX[j] += (-((m[k] - miu)**2)/(2*(sigma**2)) - np.log(sigma))
            #print(np.argmax(PYX))
            pred_labels[i] = np.argmax(PYX)
        return pred_labels



if __name__ == "__main__":

    NB = NaiveBayes()
    NB.train(train_x, train_y)

    pred = NB.predict(test_x)
    sum = 0
    for i in range(test_x.shape[0]):
        if pred[i] == test_y[i]:
           sum = sum + 1
    accuracy = sum/float(test_x.shape[0])
    print('average_accuracy = ', accuracy)

    #pickle.dump(model, open('per_models.pkl', 'wb'))
    #print('score = '+str(f1_score(test_y, pred, average='macro')))
    #print('accuracy = '+str(accuracy_score(test_y, pred)))

    print('end')

