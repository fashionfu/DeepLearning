import sympy as sp
import numpy as np
import pandas as pd
from sympy.abc import x


# 激活函数, 请使用符号变量x并返回由x构成的符号函数,以方便求导
def act_func_template():
    # 返回仅使用符号变量x的符号函数
    return x ** 1


# sigmoid激活函数
def sigmoid():
    return 1 / (1 + sp.exp(-x))


# relu激活函数
def relu():
    return x
    # return sp.Max(0, x)


# 损失函数, 请返回以列计算的大于零的array数组, 用以判断模型优劣
def loss_func_template(real_y: np.ndarray, res_y: np.ndarray):
    # 返回使用两个矩阵计算的损失函数
    return 0

# 平均值绝对值和函数(L1损失函数)
def mae(real_y: np.ndarray, res_y: np.ndarray):
    return np.sum(np.abs(res_y - real_y), axis=1) / len(real_y[0])

# 均方误差损失函数
def mse(res_y: np.ndarray, real_y: np.ndarray):
    return np.sum((res_y - real_y) ** 2, axis=1) / len(real_y[0])

# 均方根误差损失函数(L2损失函数)
def rmse(real_y: np.ndarray, res_y: np.ndarray):
    return np.sqrt(np.sum((real_y - res_y) ** 2, axis=1) / len(real_y[0]))


class BP:
    def __init__(self, lr=0.1, loss=mse, act=sigmoid, max_loss=0.0, epochs=0, batch_size=64):
        self.layers_num = 0
        self.w = None
        self.b = None
        self.dataset = None
        self.len = 0
        self.width = 0
        self.lr = lr  # 学习率
        self.loss = loss  # 损失函数
        self.loss_val = []  # 损失值(展示用)
        self.act = act()  # 激活函数
        self.err = []  # 误差(展示用)
        self.output_width = 0
        # 隐藏层形状, 不要输入输入输出层的形状
        self.shape = None
        # 二者作为fit的终止条件, epochs或max_loss为0时表示忽视该条件
        self.epochs = epochs  # 迭代次数
        self.max_loss = max_loss  # 最大损失值
        self.batch_size = batch_size  # 数据集划分大小

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        self.__shape = [self.width, *value, self.output_width] if value else [0]

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, value):
        self.__loss = value

    @property
    def act(self):
        return self.__act

    def act_g(self, _y):
        ng = self.__act_g(_y)
        if isinstance(ng, np.ndarray):
            return ng
        return np.array([ng] * _y.shape[0]).reshape(-1, 1)

    @act.setter
    def act(self, value):
        # 输入以仅x单变量的符号函数
        self.__act = sp.lambdify(x, value, 'numpy')
        self.__act_g = sp.lambdify(x, sp.diff(value, x), 'numpy')

    def __feed_forward(self, _x):
        _x = np.array(_x).reshape(-1, 1)
        for __b, __w in zip(self.b, self.w):
            _x = self.act(np.dot(__w, _x) + __b)
        return _x

    def predict(self, _dataset: np.ndarray):
        return [self.__feed_forward(__x) for __x in _dataset]

    def fit(self, shape, X, Y=None, test=None):
        # X, Y必须为二维列表且长度相等
        if Y is None:
            _dataset = pd.DataFrame(X).values
            self.len, self.width = _dataset.shape
            self.width -= 1
            self.dataset = [[np.array(_dataset[i, :-1]).reshape(-1, 1), np.array([_dataset[i, -1]]).reshape(-1, 1)] for
                            i in range(self.len)]
            self.output_width = 1
        else:
            self.dataset = [[np.array(row_x).reshape(-1, 1), np.array(row_y).reshape(-1, 1)] for row_x, row_y in
                            zip(X, Y)]
            self.len = len(self.dataset)
            self.width = len(X[0])
            self.output_width = len(Y[0])
        self.shape = shape
        self.b = [np.zeros((_h, 1)) for _h in self.shape[1:]]  # 初始b为0矩阵
        self.w = [np.random.randn(_nh, _h) for _h, _nh in zip(self.shape[:-1], self.shape[1:])]  # 初始w为随机矩阵
        self.layers_num = len(self.shape)
        self.__train(test)

    def __train(self, test=None):
        _epochs = 0
        _len = 0
        if test is not None:
            _dataset = pd.DataFrame(test).values
            _len = _dataset.shape[0]
            test = [
                [np.array(_dataset[i, :self.width]).reshape(-1, 1), np.array([_dataset[i, self.width:]]).reshape(-1, 1)]
                for i in range(_len)]
        while self.__while_check(_epochs):
            np.random.shuffle(self.dataset)
            batches = [self.dataset[k:k + self.batch_size] for k in range(0, self.len, self.batch_size)]
            loss_v = 0.0
            for batch in batches:
                loss_v += self.__train_w_b(batch)
            self.loss_val.append(loss_v / self.len)
            print(f"Epoch{_epochs + 1}/{self.epochs or 'UNKNOWN'}  loss_val:{self.loss_val[-1]}", end='')
            if test is not None:
                _test_loss_val = sum(self.loss(self.__feed_forward(tx), ty) for tx, ty in test)
                print(f" test_loss_val:{_test_loss_val / _len}", end='')
            print()
            _epochs += 1
        self.epochs = _epochs

    def __while_check(self, _epochs):
        if self.max_loss and self.epochs:
            return (not self.loss_val or any(self.loss_val[-1] >= self.max_loss)) or _epochs < self.epochs
        elif self.max_loss:
            return not self.loss_val or any(self.loss_val[-1] >= self.max_loss)
        elif self.epochs:
            return _epochs < self.epochs
        else:
            print("请设置终止条件")
            return False

    def __train_w_b(self, _batch):
        temp_b = [np.zeros(__b.shape) for __b in self.b]
        temp_w = [np.zeros(__w.shape) for __w in self.w]
        for _x, _y in _batch:
            delta_temp_b, delta_temp_w = self.__update_w_b(_x, _y)

            temp_w = [w + dw for w, dw in zip(temp_w, delta_temp_w)]
            temp_b = [b + db for b, db in zip(temp_b, delta_temp_b)]
        self.w = [sw - (self.lr / len(_batch)) * w for sw, w in zip(self.w, temp_w)]
        self.b = [sb - (self.lr / len(_batch)) * b for sb, b in zip(self.b, temp_b)]
        _loss_v = 0.0
        for _x, _y in _batch:
            _loss_v += self.loss(self.__feed_forward(_x), _y)
        return _loss_v

    def __update_w_b(self, _x, _y):
        temp_b = [np.zeros(__b.shape) for __b in self.b]
        temp_w = [np.zeros(__w.shape) for __w in self.w]
        activation = _x
        activations = [_x]
        zs = []
        for __b, __w in zip(self.b, self.w):
            z = np.dot(__w, activation) + __b
            zs.append(z)
            activation = self.act(z)
            activations.append(activation)
        d = self.__cost_derivative(activations[-1], _y) * self.act_g(zs[-1])
        temp_b[-1] = d
        temp_w[-1] = np.dot(d, activations[-2].transpose())
        for i in range(2, self.layers_num):
            z = zs[-i]
            d = np.dot(self.w[-i + 1].transpose(), d) * self.act_g(z)
            temp_b[-i] = d
            temp_w[-i] = np.dot(d, activations[-i - 1].transpose())
        return temp_b, temp_w

    def __cost_derivative(self, res_y, real_y):
        __c = res_y - real_y
        self.err.append(__c)
        return __c


if __name__ == '__main__':
    _a = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]).T
    _b = np.array([
        [1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
    ]).T
    td = [
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 0, 1, 0],
        [3, 3, 3, 0, 0, 1],
        [4, 4, 4, 1, 0, 0],
        [5, 5, 5, 0, 1, 0],
    ]
    t = BP(lr=0.7, loss=mse, act=sigmoid, epochs=0, max_loss=0.1)
    t.fit((5, 10, 5), _a, _b, test=td)
