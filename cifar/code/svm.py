# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn import svm as s
import time
import pickle


class SVM(object):
    def __init__(self):
        # 偏置权重
        self.W = None

    def svm_loss_naive(self, x, y, reg):
        """
        非矢量化版本的损失函数
        输出损失函数值loss、权重梯度dW
        """
        num_train = x.shape[0]
        num_class = self.W.shape[1]

        # 初始化
        loss = 0.0
        dW = np.zeros(self.W.shape)

        for i in range(num_train):
            scores = x[i].dot(self.W)
            # 计算边界, delta=1
            margin = scores - scores[y[i]] + 1
            # 正确
            margin[y[i]] = 0

            for j in range(num_class):
                # max
                if j == y[i]:
                    continue
                if margin[j] > 0:
                    loss += margin[j]
                    dW[:, y[i]] += -x[i]
                    dW[:, j] += x[i]

        # 除以N
        loss /= num_train
        dW /= num_train
        # 正则项
        loss += 0.5 * reg * np.sum(self.W * self.W)
        dW += reg * self.W

        return loss, dW

    def svm_loss_vectorized(self, x, y, reg):
        """
        矢量化版本的损失函数

        输出：损失函数值loss、权重梯度dW
        """
        loss = 0.0
        dW = np.zeros(self.W.shape)

        num_train = x.shape[0]
        scores = x.dot(self.W)
        margin = scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1
        margin[np.arange(num_train), y] = 0.0
        # max操作
        margin = (margin > 0) * margin
        loss += margin.sum() / num_train
        # 正则化项
        loss += 0.5 * reg * np.sum(self.W * self.W)

        # 梯度
        margin = (margin > 0) * 1
        row_sum = np.sum(margin, axis=1)
        margin[np.arange(num_train), y] = -row_sum
        dW = x.T.dot(margin) / num_train + reg * self.W

        return loss, dW

    def train(self, x, y, reg=1e-5, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False):
        """
        使用随机梯度下降法训练SVM

        """

        num_train, dim = x.shape
        num_class = np.max(y) + 1
        # 初始化权重
        if self.W is None:
            self.W = 0.005 * np.random.randn(dim, num_class)

        batch_x = None
        batch_y = None
        history_loss = []
        # 随机梯度下降法优化权重
        for i in range(num_iters):
            # 从训练样本中随机取样作为更新权重的小批量样本
            mask = np.random.choice(num_train, batch_size, replace=False)
            batch_x = x[mask]
            batch_y = y[mask]

            # 计算loss和权重的梯度
            loss, grad = self.svm_loss_vectorized(batch_x, batch_y, reg)

            # 更新权重
            self.W += -learning_rate * grad

            history_loss.append(loss)

            # 打印loss的变化过程
            if verbose == True and i % 100 == 0:
                print("iteratons:%d/%d,loss:%f" % (i, num_iters, loss))

        return history_loss

    def predict(self, x):
        """
        利用训练得到的最优权值预测分类结果
        """
        y_pre = np.zeros(x.shape[0])
        scores = x.dot(self.W)
        y_pre = np.argmax(scores, axis=1)

        return y_pre


def unpickle(file):

    """
    加载batchs.meta文件返回的字典
    """

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR10():
    """
    读取CIFAR10数据
    """

    x_t = []
    y_t = []
    for i in range(1, 6):
        path_train = os.path.join('cifar-10-batches-py', 'data_batch_%d' % (i))
        data_dict = unpickle(path_train)
        x = data_dict[b'data'].astype('float')
        y = np.array(data_dict[b'labels'])

        x_t.append(x)
        y_t.append(y)

    # 将数据按列堆叠进行合并,默认按列进行堆叠
    x_train = np.concatenate(x_t)
    y_train = np.concatenate(y_t)

    path_test = os.path.join('cifar-10-batches-py', 'test_batch')
    data_dict = unpickle(path_test)
    x_test = data_dict[b'data'].astype('float')
    y_test = np.array(data_dict[b'labels'])

    return x_train, y_train, x_test, y_test


def data_processing():
    """
    功能：进行数据预处理
    输出：
    x_tr:(numpy array)训练集数据
    y_tr:(numpy array)训练集标签
    x_val:(numpy array)验证集数据
    y_val:(numpy array)验证集标签
    x_te:(numpy array)测试集数据
    y_te:(numpy array)测试集标签
    x_check:(numpy array)用于梯度检查的子训练集数据
    y_check:(numpy array)用于梯度检查的子训练集标签
    """

    # 加载数据
    x_train, y_train, x_test, y_test = load_CIFAR10()

    num_train = 10000
    num_test = 1000
    num_val = 1000
    num_check = 100

    # 训练样本
    x_tr = x_train[0:num_train]
    y_tr = y_train[0:num_train]

    # 验证样本
    x_val = x_train[num_train:(num_train + num_val)]
    y_val = y_train[num_train:(num_train + num_val)]

    # 测试样本
    x_te = x_test[0:num_test]
    y_te = y_test[0:num_test]

    # 从训练样本中取出一个子集作为梯度检查的数据
    mask = np.random.choice(num_train, num_check, replace=False)
    x_check = x_tr[mask]
    y_check = y_tr[mask]

    # 均值
    mean_img = np.mean(x_tr, axis=0)

    x_tr += -mean_img
    x_val += -mean_img
    x_te += -mean_img
    x_check += -mean_img

    x_tr = np.hstack((x_tr, np.ones((x_tr.shape[0], 1))))
    x_val = np.hstack((x_val, np.ones((x_val.shape[0], 1))))
    x_te = np.hstack((x_te, np.ones((x_te.shape[0], 1))))
    x_check = np.hstack((x_check, np.ones((x_check.shape[0], 1))))

    return x_tr, y_tr, x_val, y_val, x_te, y_te, x_check, y_check


def VisualizeWeights(best_W):
    # 去除最后一行偏置项
    w = best_W[:-1, :]
    w = w.T
    w = np.reshape(w, [10, 3, 32, 32])
    # 对应类别
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        plt.subplot(2, 5, i + 1)
        # 将图像拉伸到0-255
        x = w[i]
        minw, maxw = np.min(x), np.max(x)
        wimg = (255 * (x.squeeze() - minw) / (maxw - minw)).astype('uint8')

        r = Image.fromarray(wimg[0])
        g = Image.fromarray(wimg[1])
        b = Image.fromarray(wimg[2])

        wimg = Image.merge("RGB", (r, g, b))

        plt.imshow(wimg)
        plt.axis('off')
        plt.title(classes[i])


# 主函数
if __name__ == '__main__':
    # 进行数据预处理
    x_train, y_train, x_val, y_val, x_test, y_test, x_check, y_check = data_processing()

    start = time.clock()
    # 调参
    # learning_rate = [7e-6, 1e-7, 3e-7]
    # regularization_strength = [1e4, 3e4, 5e4, 7e4, 1e5, 3e5, 5e5]

    # learning_rate = [1e-3, 3e-3, 6e-3, 9e-3, 1e-4, 3e-4, 6e-4, 9e-4, 1e-5, 3e-5, 6e-5, 9e-5, \
    #                  1e-6, 3e-6, 6e-6, 9e-6, 1e-7, 3e-7, 6e-7, 9e-7, 1e-8, 3e-8, 6e-8, 9e-8]
    # regularization_strength = [1e2, 3e2, 6e2, 9e2, 1e3, 3e3, 6e3, 9e3, 1e4, 3e4, 6e4, 9e4,  \
    #                            1e5, 3e5, 6e5, 9e5, 1e6, 3e6, 6e6, 9e6, 1e7, 3e7, 6e7, 9e7]

    # learning_rate = [1e-4, 3e-4, 1e-5, 3e-5, 1e-6, 3e-6, 1e-7, 3e-7, 1e-8, 3e-8]
    # regularization_strength = [1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]

    learning_rate = [4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-7, 2e-7, 3e-7]
    regularization_strength = [1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]

    l = []
    r = []
    a = []

    max_acc = -1.0
    for lr in learning_rate:
        for rs in regularization_strength:
            svm = SVM()
            # 训练
            history_loss = svm.train(x_train, y_train, reg=rs, learning_rate=lr, num_iters=2000)
            # 预测验证集类别
            y_pre = svm.predict(x_val)
            # 计算验证集精度
            acc = np.mean(y_pre == y_val)

            l.append(lr)
            r.append(rs)
            a.append(acc)

            # 选取精度最大时的最优模型
            if (acc > max_acc):
                max_acc = acc
                best_learning_rate = lr
                best_regularization_strength = rs
                best_svm = svm

            print("learning_rate=%e,regularization_strength=%e,val_accury=%f" % (lr, rs, acc))
    print("max_accuracy=%f,best_learning_rate=%e,best_regularization_strength=%e" % (
    max_acc, best_learning_rate, best_regularization_strength))
    end = time.clock()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(l, r, a)
    ax.legend()
    plt.show()

    # 用最优svm模型对测试集进行分类的精度
    # 预测测试集类别
    y_pre = best_svm.predict(x_test)
    # 计算测试集精度
    acc = np.mean(y_pre == y_test)
    print('The test accuracy with self-realized svm is:%f' % (acc))
    print("\nProgram time of self-realized svm is:%ss" % (str(end - start)))

    # 可视化学习到的权重
    VisualizeWeights(best_svm.W)

    start = time.clock()
    lin_clf = s.LinearSVC()
    lin_clf.fit(x_train, y_train)
    y_pre = lin_clf.predict(x_test)
    acc = np.mean(y_pre == y_test)
    print("The test accuracy with svm.LinearSVC is:%f" % (acc))
    end = time.clock()
    print("Program time of svm.LinearSVC is:%ss" % (str(end - start)))