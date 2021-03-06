
#perceptron感知机

import numpy as np
import pickle
# 绘制模型效果折线图
import matplotlib.pyplot as plt
#引入评估函数
from sklearn.metrics import accuracy_score, f1_score
#加载数据集
from sklearn.datasets import fetch_20newsgroups
#挑选部分数据集来进行测试
categories = ['alt.atheism', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'talk.religion.misc']

#训练集
#newsgroups_train = fetch_20newsgroups(subset='train',categories=categories,remove='header')
newsgroups_train = fetch_20newsgroups(subset='train', remove='header')
#测试集
#newsgroups_test = fetch_20newsgroups(subset='test',categories=categories,remove='header')
newsgroups_test = fetch_20newsgroups(subset='test', remove='header')


#导入本地停用词表
with open("D:\大二下_课程资料\机器学习实验\skl\stopwords.txt", "rb") as f:
    stpwrdlst = f.read()
#将文本转为TF-IDF向量
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# 停用词为stopwords.txt，全部转换为小写，选择词频为前5000的作为特征，构造稀疏矩阵




class perceptron:

    def __init__(self, learningRate=1):
        # learningRate是学习率，即是每次更新权重和截距的步长
        self.learningRare = learningRate
        self.w = 0
        self.b = 0

    def train(self, data, target):
        row = data.shape[0]
        col = data.shape[1]
        self.w = np.zeros(col)
        #print(row)
        #print(col)
        # indptr表示矩阵中每一行的数据在data中开始和结束的索引
        # indices中表示所对应的在data中的数据在矩阵中其所在行的所在列数。
        #print(data.indptr)
        #print(data.indices)
        print('Start training')
        # 对稀疏矩阵进行解压缩变换得到一行向量
        for i in range(row):
            m = np.zeros(col)
            bound_0 = data.indptr[i]
            bound_1 = data.indptr[i+1]
            for j in range(bound_0, bound_1):
                #print(j)
                m[data.indices[j]] = data.data[j]
                #print('indices = '+str(data.indices[j]))

            # 梯度下降法
            result = (np.dot(self.w, m) + self.b) * target[i]
            #print("计算之后的结果为：")
            #print(result)
            if result <= 0:
                # print('分类错误')
                self.w = self.w + target[i] * m * self.learningRare
                self.b = self.b + target[i] * self.learningRare
                #print("调整之后的参数为：")
                # print(self.w)
                #print(self.b)

        #print("最终的参数")
        #print(self.w)
        #print(self.b)

    def predict(self, data):
        print('Start predicting')
        pred_labels = []
        row = data.shape[0]
        col = data.shape[1]
        for i in range(row):
            m = np.zeros(col)
            bound_0 = data.indptr[i]
            bound_1 = data.indptr[i+1]
            for j in range(bound_0, bound_1):
                m[data.indices[j]] = data.data[j]
            # 分类预测
            result = np.dot(self.w, m) + self.b
            if result > 0:
                pred_labels.append(1)
            else:
                pred_labels.append(-1)
        return pred_labels



if __name__ == "__main__":


    avg_f1 = [] #保存不同特征个数的f1
    avg_accuracy = [] #保存不同特征个数的accuracy
    # 设置不同特征个数训练模型
    features = [1000, 2000, 3000, 5000, 10000, 20000,30000]
    #print('class_num = ' +str(len(classes)))
    for feature_ in features:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, lowercase=True, max_features=feature_)
        # 训练集对应稀疏矩阵
        vectors_train = vectorizer.fit_transform(newsgroups_train.data)
        # 测试集对应稀疏矩阵
        vectors_test = vectorizer.transform(newsgroups_test.data)
        # 训练集转换后的矩阵大小
        print(vectors_train.shape)
        # 测试集转换后的矩阵大小
        #print(vectors_test.shape)
        # 非零特征的个数
        #print(vectors_train.nnz / float(vectors_train.shape[0]))
        #print(vectors_test.nnz / float(vectors_test.shape[0]))
        train_x = vectors_train  # 训练集数据
        train_y = newsgroups_train.target  # 训练集标签
        test_x = vectors_test  # 测试集数据
        test_y = newsgroups_test.target  # 测试集标签
        model = []  # 保存感知机
        score = {}  # 保存每个感知机得分
        accuracy = {}  # 保存每个感知机准确率
        classes = np.unique(train_y)  # 去掉多余项，得到分类的种类数
        # print(classes)
        class_num = len(classes)
        for i in range(class_num): # 训练多个感知机
            copy_train_y = train_y.copy()
            copy_test_y = test_y.copy()
            for j in range(copy_train_y.shape[0]): #对每一个分类统一化使用 one VS rest方法
                if copy_train_y[j] == i:    # 符合该分类为1
                    copy_train_y[j] = 1
                else:                       # 非该分类为-1
                    copy_train_y[j] = -1
            for j in range(copy_test_y.shape[0]):
                if copy_test_y[j] == i:
                    copy_test_y[j] = 1
                else:
                    copy_test_y[j] = -1
            print('epoch : '+str(i+1))
            model = perceptron(learningRate=1)
            model.train(train_x, copy_train_y)
            pred = model.predict(test_x)
            #计算得分与准确率
            score[i] = f1_score(copy_test_y, pred, average='macro')
            accuracy[i] =accuracy_score(copy_test_y, pred)


    #保存模型、得分、准确率
    #pickle.dump(model, open('per_models.pkl', 'wb'))
    #pickle.dump(score,open('per_score.pkl','wb'))
    #pickle.dump(accuracy, open('per_accurarcy.pkl', 'wb'))

    # 计算平均评价指标
        sum_score = 0
        sum_accuracy= 0
        for i in range(class_num):
            sum_score = sum_score + score[i]
            sum_accuracy = sum_accuracy + accuracy[i]
        #print('average_score = '+str(sum_score/float(class_num)))
        #print('average_accuracy = '+str(sum_accuracy/float(class_num)))
        avg_f1.append(sum_score/float(class_num))
        avg_accuracy.append(sum_accuracy/float(class_num))

    pickle.dump(features, open('p_features.pkl', 'wb'))
    pickle.dump(avg_f1, open('p_f1.pkl', 'wb'))
    pickle.dump(avg_accuracy, open('p_acc.pkl', 'wb'))
    plt.plot(avg_f1)
    plt.plot(avg_accuracy)
    plt.ylim(ymin=0.6, ymax=1.01)
    plt.title("The score of perceptron model")
    plt.legend(["f1_score", "accuracy"])
    plt.show()


    '''
    #测试单个感知机
    model = perceptron(learningRate=1)
    model.train(train_x, train_y)
    pred = model.predict(test_x)
    print(pred)
    #pickle.dump(model, open('per_models.pkl', 'wb'))
    #print('score = '+str(f1_score(test_y, pred, average='macro')))
    #print('accuracy = '+str(accuracy_score(test_y, pred)))
    '''
    print('end')

