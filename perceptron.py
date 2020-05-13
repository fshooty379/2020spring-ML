
#perceptron感知机

import numpy as np
import pickle
from sklearn.metrics import accuracy_score,f1_score
#加载数据集
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'talk.religion.misc', 'sci.space']
#训练集
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
#newsgroups_train = fetch_20newsgroups(subset='train')
#测试集
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
#newsgroups_test = fetch_20newsgroups(subset='test')


#导入本地停用词表
with open("D:\大二下_课程资料\机器学习实验\skl\stopwords.txt", "rb") as f:
    stpwrdlst = f.read()
#将文本转为TF-IDF向量
from sklearn.feature_extraction.text import TfidfVectorizer
# 停用词为stopwords.txt，全部转换为小写，提取tfidf特征（词频、逆文档频率）应用于稀疏矩阵
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,lowercase=True,max_features = 1000)
#转换为tf-idf向量之后的数据
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
# 训练集转换后的向量大小
print(vectors_train.shape)
# 测试集转换后的向量大小
print(vectors_test.shape)
# 非零特征的个数
print(vectors_train.nnz / float(vectors_train.shape[0]))
print(vectors_test.nnz / float(vectors_test.shape[0]))

train_x = vectors_train #训练集数据
train_y = newsgroups_train.target # 训练集标签
test_x = vectors_test # 测试集数据
test_y = newsgroups_test.target #测试集标签



class perceptron:
    '''learningRate是学习率，即是每次更新权重和截距的步长'''

    def __init__(self, learningRate=1):
        self.learningRare = learningRate
        self.w = 0
        self.b = 0

    def train(self, data, target):
        row = data.shape[0]
        col = data.shape[1]
        self.w = np.zeros([1, col])
        #print(row)
        #print(col)
        # indptr表示矩阵中每一行的数据在data中开始和结束的索引
        # indices中表示所对应的在data中的数据在矩阵中其所在行的所在列数。
        #print(data.indptr)
        #print(data.indices)
        print('Start training')
        # 对样本进行变换得到一列列向量
        for i in range(row):
            m = np.zeros([col,1])
            bound = data.indptr[i+1]
            #print(bound)
            j = 0
            for j in range(bound):
                #print(j)
                m[data.indices[j]] = data.data[j]
                #print('indices = '+str(data.indices[j]))

            # 梯度下降进行计算
            result = (np.dot(self.w, m) + self.b) * target[i]
            print("计算之后的结果为：")
            print(result)

            if result <= 0:
                # print('分类错误')
                self.w = self.w + target[i] * data[i] * self.learningRare
                self.b = self.b + target[i] * self.learningRare
                print("调整之后的参数为：")
                # print(self.w)
                print(self.b)


        print("最终的参数")
        print(self.w)
        print(self.b)

    def predict(self,data):
        print('Start predicting')
        pred_labels = []
        row = data.shape[0]
        col = data.shape[1]
        for i in range(row):
            m = np.zeros([col,1])
            bound = data.indptr[i+1]
            j = 0
            for j in range(bound):
                m[data.indices[j]] = data.data[j]

            result = np.dot(self.w,m) + self.b
            if result > 0:
                pred_labels.append(1)
            else:
                pred_labels.append(-1)
        return pred_labels



if __name__ == "__main__":

    model = []
    score = {}
    accuracy = {}
    for i in range(3):
        copy_train_y = train_y
        copy_test_y = test_y
        for j in range(train_y.shape[0]):
            if copy_train_y[j] == i:
                copy_train_y[j] = 1
            else:
                copy_train_y[j] = -1
        for j in range(test_y.shape[0]):
            if copy_test_y[j] == i:
                copy_test_y[j] = 1
            else:
                copy_test_y[j] = -1
        model = perceptron(learningRate=1)
        model.train(train_x, copy_train_y)
        pred = model.predict(test_x)
        score[i] = f1_score(copy_test_y, pred, average='macro')
        accuracy[i] =accuracy_score(copy_test_y, pred)
    pickle.dump(model, open('per_models.pkl', 'wb'))
    pickle.dump(score,open('per_score.pkl','wb'))
    pickle.dump(accuracy, open('per_accurarcy.pkl', 'wb'))
    sum_score = 0
    sum_accuracy= 0
    for i in range(3):
        sum_score = sum_score + score[i]
        sum_accuracy = sum_accuracy + accuracy[i]
    print('score = '+str(sum_score/3.0))
    print('accuracy = '+str(sum_accuracy/3.0))

    '''
    model = perceptron(learningRate=1)
    model.train(train_x, train_y)
    pred = model.predict(test_x)
    pickle.dump(model, open('per_models.pkl', 'wb'))
    '''
    #print('score = '+str(f1_score(test_y, pred, average='macro')))
    #print('accuracy = '+str(accuracy_score(test_y, pred)))
    print('end')

