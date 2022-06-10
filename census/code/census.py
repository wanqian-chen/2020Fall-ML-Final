import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 绘图函数
def figure(title, *datalist):
    plt.figure(facecolor='gray', figsize=[16, 8])
    for v in datalist:
        plt.plot(v[0], '-', label=v[1], linewidth=2)
        plt.plot(v[0], 'o')
    plt.grid()
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)
    plt.show()


def Print(y_train_pred, y_test_pred, y_train, y_test, method):

    # 绘制预测值与真实值图
    figure(method + ' 预测值与真实值图 模型的' + r'$R^2=%.4f$' % (r2_score(y_train_pred, y_train)), [y_test_pred, '预测值'],
           [y_test, '真实值'])


data = pd.read_csv("近20年人口普查数据.csv",encoding='gbk')

# 从小到大排序
data = data.sort_index(ascending=0)

census_df = pd.DataFrame(data, columns=['指标','年末总人口(万人)','男性人口(万人)','女性人口(万人)','城镇人口(万人)','乡村人口(万人)'])
# XA = np.array(['2020'],['2021'],['2022'],['2023'],['2024'],['2025'],dtype=object)

plt.figure()
plt.plot(census_df['指标'], census_df['年末总人口(万人)'],'k',\
         census_df['指标'], census_df['男性人口(万人)'],'ro-',\
         census_df['指标'], census_df['女性人口(万人)'],'bs--',\
         census_df['指标'], census_df['城镇人口(万人)'],'yH-',\
         census_df['指标'], census_df['乡村人口(万人)'],'gH--')
plt.legend(['年末总人口(万人)','男性人口(万人)','女性人口(万人)','城镇人口(万人)','乡村人口(万人)'])
plt.show()

Y = [0,0,0,0,0]
Y[0] = np.array(census_df['年末总人口(万人)'])
Y[1] = np.array(census_df['男性人口(万人)'])
Y[2] = np.array(census_df['女性人口(万人)'])
Y[3] = np.array(census_df['城镇人口(万人)'])
Y[4] = np.array(census_df['乡村人口(万人)'])
# 特征值
X = np.array(census_df['指标']).reshape(-1, 1)


i = -1
# 测试集占30%
for y in Y:
    i += 1
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 归一化
    stand = StandardScaler()
    x_train = stand.fit_transform(x_train)
    x_test = stand.fit_transform(x_test)


    # 多项式
    poly_reg = PolynomialFeatures(degree=2)
    x_train_poly = poly_reg.fit_transform(x_train)
    x_test_poly = poly_reg.fit_transform(x_test)

    lr = LinearRegression()
    lr.fit(x_train_poly, y_train) # 训练数据
    y_train_pred = lr.predict(x_train_poly) # 预测数据
    y_test_pred = lr.predict(x_test_poly)

    Print(y_train_pred, y_test_pred, y_train, y_test, '多项式, y[%s]' % i)


    # 线性模型
    lr = LinearRegression()
    lr.fit(x_train, y_train) # 训练数据
    y_train_pred = lr.predict(x_train) # 预测数据
    y_test_pred = lr.predict(x_test)
    # print(y_predict)
    score1 = lr.score(x_test,y_test) # 准确率
    weight1 = lr.coef_ # 权重
    bias1 = lr.intercept_ # 偏置
    # show_res(y_test,y_predict)

    print(score1)
    print(weight1)
    print(bias1)
    Print(y_train_pred, y_test_pred, y_train, y_test, '线性, y[%s]' % i)

    # +L2正则化
    rd = Ridge()
    rd.fit(x_train, y_train) # 训练数据
    y_train_pred = rd.predict(x_train) # 预测数据
    y_test_pred = rd.predict(x_test)
    # print(y_predict)
    score2 = rd.score(x_test,y_test) # 准确率
    weight2 = rd.coef_ # 权重
    bias2 = rd.intercept_ # 偏置
    # show_res(y_test,y_predict)
    
    print(score2)
    print(weight2)
    print(bias2)
    Print(y_train_pred, y_test_pred, y_train, y_test, '+L2正则化, y[%s]' % i)
    
    # 4折
    # 多项式
    scores = cross_val_score(estimator=lr.fit(x_train_poly, y_train), X=x_train, y=y_train, cv=4, n_jobs=1)
    print('y[%s]多项式准确率：%s' % (i, scores))
    print('y[%s]多项式平均准确率：%s' % (i, np.mean(scores)))
    # 线性
    scores = cross_val_score(estimator=lr.fit(x_train,y_train), X=x_train, y=y_train, cv=4, n_jobs=1)
    print('y[%s]线性准确率：%s' % (i, scores))
    print('y[%s]线性平均准确率：%s' % (i, np.mean(scores)))
    # +L2正则化
    scores = cross_val_score(estimator=rd.fit(x_train, y_train), X=x_train, y=y_train, cv=4, n_jobs=1)
    print('y[%s]+L2正则化准确率：%s' % (i, scores))
    print('y[%s]+L2正则化平均准确率：%s' % (i, np.mean(scores)))
