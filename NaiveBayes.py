#NaiveBayes朴素贝叶斯

import numpy as np
import pickle
from sklearn.metrics import accuracy_score,f1_score
from math import pi, exp
from scipy.stats import norm
#加载数据集
from sklearn.datasets import fetch_20newsgroups
#挑选部分数据集来进行bug测试
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
              'talk.religion.misc', 'sci.electronics']
              #'rec.sport.baseball', 'sci.space', 'talk.politics.guns', 'sci.med']
#训练集
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove='header')
#newsgroups_train = fetch_20newsgroups(subset='train',remove='header')
#测试集
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories,remove='header')
#newsgroups_test = fetch_20newsgroups(subset='test',remove='header')


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
print(vectors_train.nnz / float(vectors_train.shape[0]))
print(vectors_test.nnz / float(vectors_test.shape[0]))

train_x = vectors_train #训练集数据
train_y = newsgroups_train.target # 训练集标签
test_x = vectors_test # 测试集数据
test_y = newsgroups_test.target #测试集标签
sqrt_pi = (2 * pi) ** 0.5

class NaiveBayes:
    def __init__(self,class_num):

        self.prior = np.zeros(class_num) # 存储先验概率
        self.avgs = np.zeros((class_num, 10000))  # 存储训练集均值
        self.vars = np.zeros((class_num, 10000)) # 存储方差
        self.n_class = class_num # 存储类别数量



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
        # print(self.prior)
        # 计算每一个类下的训练集均值与方差
        for i in range(self.n_class):
            m = [] # 存放当前样本
            print('类别 ', i)
            sum = 0
            for j in range(row): # max_j = 1655
                if target[j] == i: #属于当前类别
                    sum = sum + 1
                    tmp = np.zeros(col)
                    for k in range(data.indptr[i], data.indptr[i+1]):
                        tmp[data.indices[k]] = data.data[k]
                    #print(tmpm.shape)
                    m.append(tmp)
                    #print(len(m))
            print('sum = ', sum)
            m = np.array(m).T
            #print(m)
            #print('len = ', len(m))
            #print(np.mean(m[j]))
            #exit()
            for j in range(col): #max_j = 2000
                self.avgs[i][j] = np.mean(m[j])
                self.vars[i][j] = np.var(m[j])
        #print(self.vars.shape)
        #print(self.avgs[0])

    # 高斯概率密度函数
    def gaussian(self, x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma)) / (sqrt_pi * sigma ** 0.5)

    '''
    def predict(self, data):
        eps = 1e-4
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
            PYX = np.zeros(self.n_class) # 存储P（Y|X）
            for j in range(self.n_class):
                PYX[j] = np.log(self.prior[j])
                for k in range(self.vars.shape[1]):
                    mu = self.avgs[j][k]
                    sigma = self.vars[j][k]
                    if mu*sigma == 0:
                        continue
                   # PYX[j] += self.Gaussian_probability(m, miu, sigma)
                   # PYX[j] += (-((m[k] - mu)**2)/(2*(sigma**2)) - np.log(sigma))
                    PYX[j] += self.gaussian(m[k], mu, sigma)
            #print(np.argmax(PYX))
            pred_labels[i] = np.argmax(PYX)
        return pred_labels
    '''
    def _pdf(self, x, classes):
        # 用高斯分布拟合p(x|y),也就是后验概率.并且按行每个特征的p(x|y)累乘,取log成为累加.
        eps = 1e-4  # 防止分母为0
        #mean = self.parameters['class' + str(classes)]['mean']
        mean = self.avgs[classes]
        print(mean)
        #var = self.parameters['class' + str(classes)]['var']
        var = self.vars[classes]
        print(var)
        #分子
        fenzi = np.exp(-(x - mean) ** 2 / (2 * (var) ** 2 + eps))
        #分母
        fenmu = (2 * np.pi) ** 0.5 * var + eps
        result = np.sum(np.log(fenzi / fenmu), axis=1, keepdims=True)
        return result.T

    #计算每一个种类的P(y)P(x|y)
    def _predict(self, x):
        output = []
        for y in range(self.n_class):
            prior = np.log(self.prior[y])
            posterior = self._pdf(x, y)
            prediction = prior + posterior
            output.append(prediction)
        return output

    def predict_1(self, x):
        output = self._predict(x)
        output = np.reshape(output, (self.n_class, x.shape[0]))
        prediction  = np.argmax(output, axis=0)
        return prediction

if __name__ == "__main__":

    classes = np.unique(train_y)  # 去掉多余项，得到分类的种类数
    class_nums = len(classes)
    NB = NaiveBayes(class_num = class_nums)
    #训练
    NB.train(train_x, train_y)
    #预测
    pred = NB.predict_1(test_x)
    '''
    #计算准确率
    sum = 0
    for i in range(test_x.shape[0]):
        if pred[i] == test_y[i]:
           sum = sum + 1
    accuracy = sum/float(test_x.shape[0])
    print('average_accuracy = ', accuracy)
    '''
    print("朴素贝叶斯准确率：\n", accuracy_score(test_y, pred))
    #pickle.dump(model, open('per_models.pkl', 'wb'))
    #print('score = '+str(f1_score(test_y, pred, average='macro')))
    #print('accuracy = '+str(accuracy_score(test_y, pred)))

    print('end')



'''
class NaiveBayes:

    def fit(self, x, y, class_num):
        self.x = x
        self.y = y
        self.classes = np.unique(y)



        self.parameters = {}
        for i, c in enumerate(self.classes):
            # 计算属于同一类别的均值,方差和各类别的先验概率p(y).
            X_index_c = x[np.where(y == c)]
            X_index_c_mean = np.mean(X_index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_index_c, axis=0, keepdims=True)
            parameters = {'mean': X_index_c_mean, 'var': X_index_c_var, 'prior': X_index_c.shape[0] / x.shape[0]}
            #parameters = {'mean': self.avgs[i], 'var': self.vars[i], 'prior': X_index_c.shape[0] / x.shape[0]}
            self.parameters['class' + str(c)] = parameters  # 字典嵌套
            print('prior :')
            print(parameters['prior'])
    #             print(X_index_c.shape[0])

    def _pdf(self, x, classes):
        # 用高斯分布拟合p(x|y),也就是后验概率.并且按行每个特征的p(x|y)累乘,取log成为累加.
        eps = 1e-10  # 防止分母为0
        mean = self.parameters['class' + str(classes)]['mean']
        var = self.parameters['class' + str(classes)]['var']
        fenzi = np.exp(-(x - mean) ** 2 / (2 * (var) ** 2 + eps))
        fenmu = (2 * np.pi) ** 0.5 * var + eps
        result = np.sum(np.log(fenzi / fenmu), axis=1, keepdims=True)
        # print(result.T.shape)
        return result.T  # (1, 719)

    def _predict(self, x):
        # 计算每个种类的p(y)p(x|y)
        output = []
        for y in range(self.classes.shape[0]):
            prior = np.log(self.parameters['class' + str(y)]['prior'])
            posterior = self._pdf(x, y)
            prediction = prior + posterior
            output.append(prediction)
        return output

    def predict(self, x):
        # argmax(p(y)p(x|y))就是最终的结果
        output = self._predict(x)
        output = np.reshape(output, (self.classes.shape[0], x.shape[0]))
        prediction = np.argmax(output, axis=0)
        return prediction


#训练
classes = np.unique(train_y)  # 去掉多余项，得到分类的种类数
class_nums = len(classes)
#NB = NaiveBayes(class_num = class_nums)
train_tfidf_matrix_1 = train_x.toarray()
mnb = NaiveBayes()
mnb.fit(train_tfidf_matrix_1, train_y,class_nums)
#预测
test_tfidf_matrix1 = test_x.toarray()
pred_test_label_num = mnb.predict(test_tfidf_matrix1)
#混淆矩阵输出
#c_m = confusion_matrix(test_label_num, pred_test_label_num, labels=None, sample_weight=None)
#print("原始贝叶斯混淆矩阵:\n",c_m)
print("原始贝叶斯准确率：\n",accuracy_score(test_y, pred_test_label_num))
'''
