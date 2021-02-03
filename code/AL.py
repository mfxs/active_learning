# 导入库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# 预测函数
def predict_model(x):
    y = -0.85 * np.cos(3 * x) * x * np.exp(-(0.8 * x - 0.4) ** 2)
    return y


# 产生所有数据
n = 100000
x = np.linspace(-7, 7, n)
y = np.array(list(map(predict_model, x)))

# 初始化训练集
num_init = int(input('Input the number of initial samples:'))
index_init = list(np.linspace(0, n - 1, num_init).astype(int))

# 选择主动学习策略
print(
    'Active learning strategies list:\n1-Relative variance(only on train set)\n2-Variance(only on train set)\n3-Relative variance(on dataset)\n4-Variance(on dataset)')
method = int(input('Input active learning strategy:'))

# 主动学习循环
index = index_init
for i in range(200):
    print(index)
    x_avail = x[index]
    y_avail = y[index]
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    reg = GaussianProcessRegressor(alpha=0.001, kernel=kernel).fit(np.atleast_2d(x_avail).T, y_avail)
    y_mean, y_std = reg.predict(np.atleast_2d(x).T, return_std=True)
    y_var = y_std ** 2

    # 每增加10个训练数据画一次图
    if i % 10 == 0:
        plt.fill_between(x, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, color='darkorange', label='95%')
        plt.plot(x, y, label='truth')
        plt.plot(x, y_mean, 'g', label='predict')
        plt.plot(x_avail, y_avail, 'r.', label='train_data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()
        plt.title('Iteration ' + str(i))
        plt.show()

    # 寻找训练样本中相对预测方差最大的样本点
    if method == 1:
        var_avail = y_var[index]
        index_max = np.abs(np.diff(var_avail)).argmax()
        index.append(int((index[index_max] + index[index_max + 1]) / 2))

    # 寻找训练样本中预测方差最大的样本点
    elif method == 2:
        width = 500
        index_max = y_var[index].argmax()
        while 1:
            index_add = np.random.randint(max(index[index_max] - width, 0), min(index[index_max] + width + 1, n))
            if index_add != index_max:
                index.append(index_add)
                break

    # 寻找所有样本中相对预测方差最大的样本点
    elif method == 3:
        index_max = np.abs(np.diff(y_var)).argmax()
        index.append(index_max + 1)

    # 寻找所有样本中预测方差最大的样本点
    elif method == 4:
        index_max = y_var.argmax()
        index.append(index_max)

    # 训练样本依据大小重新排序
    index.sort()
