#perceptron感知机

import numpy as np
import pickle
#引入评估函数
from sklearn.metrics import accuracy_score,f1_score
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
from sklearn.feature_extraction.text import TfidfVectorizer
# 停用词为stopwords.txt，全部转换为小写，选择词频为前5000的作为特征，构造稀疏矩阵
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,lowercase=True,max_features=10000)
#训练集对应稀疏矩阵
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
#测试集对应稀疏矩阵
vectors_test = vectorizer.transform(newsgroups_test.data)
# 训练集转换后的矩阵大小
print(vectors_train.shape)
# 测试集转换后的矩阵大小
print(vectors_test.shape)
# 非零特征的个数
#print(vectors_train.nnz / float(vectors_train.shape[0]))
#print(vectors_test.nnz / float(vectors_test.shape[0]))

train_x = vectors_train #训练集数据
train_y = newsgroups_train.target # 训练集标签
test_x = vectors_test # 测试集数据
test_y = newsgroups_test.target #测试集标签



class perceptron:

    def __init__(self, learningRate, epoches):
        # learningRate是学习率，即是每次更新权重和截距的步长
        # epoches是训练次数
        self.learningRare = learningRate
        self.w = 0
        self.b = 0
        self.epoch = epoches

    def train(self, data, target):
        for a in range(self.epoch):
            row = data.shape[0]
            col = data.shape[1]
            self.w = np.zeros(col)
            # indptr表示矩阵中每一行的数据在data中开始和结束的索引
            # indices中表示所对应的在data中的数据在矩阵中其所在行的所在列数。
            print('Start training')
            # 对稀疏矩阵进行解压缩变换得到一行向量
            for i in range(row):
                m = np.zeros(col)
                bound_0 = data.indptr[i]
                bound_1 = data.indptr[i+1]
                for j in range(bound_0, bound_1):
                    m[data.indices[j]] = data.data[j]
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

def predict_final(data, w, b):
    print('Start predicting')
    pred_labels = np.zeros(7532)
    row = data.shape[0]
    col = data.shape[1]
    right = []  # 保存分离超平面时结果为正的分类
    for i in range(row):
        lens = np.zeros(20)
        m = np.zeros(col)
        bound_0 = data.indptr[i]
        bound_1 = data.indptr[i + 1]
        for j in range(bound_0, bound_1):
            m[data.indices[j]] = data.data[j]
        # 分类预测
        for k in range(20):  # 20组w和b
            result = np.dot(w[k], m) + b[k]
            if result > 0:  # 如果分离超平面结果为正
                right.append(k)  # 存入初步正确的分类数组中，k为类别号
                norm = np.linalg.norm(w[k], ord=2, keepdims=True) #L2范数
                lens[k] = (result / norm) #求误分类点到超平面的距离
        # print(right) #初步判断文本属于哪一类
        max_lens = max(lens) #找出最大距离
        # print(max_lens)
        for z in range(lens.shape[0]):
            if lens[z] == max_lens:
                pred_labels[i] = z  #得到最大距离对应的类别
    return pred_labels

if __name__ == "__main__":

    classes = np.unique(train_y) # 去掉多余项，得到分类的种类数
    class_num = len(classes)
    # 保存每个感知机的w和b
    w =[]
    b = []
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
        print('第'+str(i+1)+'类')
        model = perceptron(learningRate=1, epoches=3)
        model.train(train_x, copy_train_y)
        w.append(model.w)
        b.append(model.b)

    #print(w)
    #print(b)
    pred = predict_final(test_x, w, b)
    pred = np.array(pred)
    #print(pred)
    #print(test_y)
    print('perceptron_f1_score = '+str(f1_score(test_y, pred, average='macro')))
    print('perceptron_accuracy = '+str(accuracy_score(test_y, pred)))

    '''
    #测试单个感知机
    model = perceptron(learningRate=1)
    model.train(train_x, train_y)
    pred = model.predict(test_x)
    pickle.dump(model, open('per_models.pkl', 'wb'))
    #print('score = '+str(f1_score(test_y, pred, average='macro')))
    #print('accuracy = '+str(accuracy_score(test_y, pred)))
    '''
    print('end')