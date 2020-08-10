# %%
import numpy as np
import pandas as pd

#%%
# sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 수치미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


#%%
class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.gate_name = gate_name
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        self.__w = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        self.__learning_rate = 1e-1

    def __loss_func(self):
        delta = 1e-7  # log 무한대 발산 방지

        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)

        # cross-entropy
        return -np.sum(
            self.__tdata * np.log(y + delta)
            + (1 - self.__tdata) * np.log((1 - y) + delta)
        )

    def train(self):
        f = lambda x: self.__loss_func()

        print("Initial error value = ", self.error_val())

        for step in range(8001):

            self.__w -= self.__learning_rate * numerical_derivative(f, self.__w)

            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)

            if step % 400 == 0:
                print("step = ", step, "error value = ", self.error_val())

    def error_val(self):
        delta = 1e-7  # log 무한대 발산 방지
        z = np.dot(self.__xdata, self.__w) + self.__b
        y = sigmoid(z)
        # cross-entropy
        return -np.sum(
            self.__tdata * np.log(y + delta)
            + (1 - self.__tdata) * np.log((1 - y) + delta)
        )

    def predict(self, input_data):
        z = np.dot(input_data, self.__w) + self.__b
        y = sigmoid(z)

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False

        return y, result


# %%
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 0, 0, 1])
AND_obj = LogicGate("AND_GATE", xdata, tdata)

# %%
AND_obj.train()


# %%
test_case = [[0, 0], [0, 1], [1, 0], [1, 1]]
for test in test_case:
    sig, logical = AND_obj.predict(test)
    print(sig, logical)

# %%

xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 1, 1, 1])
OR_obj = LogicGate("OR_GATE", xdata, tdata)

# %%
OR_obj.train()


# %%
test_case = [[0, 0.49], [0, 1], [1, 0], [1, 1]]
for test in test_case:
    sig, logical = OR_obj.predict(test)
    print(sig, logical)

# %%

xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([1, 1, 1, 0])
NAND_obj = LogicGate("NAND_GATE", xdata, tdata)

# %%
NAND_obj.train()


# %%
test_case = [[0, 0], [0, 1], [1, 0], [1, 1]]
for test in test_case:
    sig, logical = NAND_obj.predict(test)
    print(sig, logical)


# %%
# XOR = NANA + OR
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

s1 = []  # NAND
s2 = []  # OR

final_output = []  # AND 출력

for index in range(len(input_data)):
    s1 = NAND_obj.predict(input_data[index])  # NANA 출력
    s2 = OR_obj.predict(input_data[index])  # OR 출력
    new_input_data = []  # AND 입력

    new_input_data.append(s1[-1])  # AND 입력
    new_input_data.append(s2[-1])  # AND 입력

    (sig, logi) = AND_obj.predict(np.array(new_input_data))
    final_output.append(logi)

for index in range(len(input_data)):
    print(f"{input_data[index]} = {final_output[index]} \n")


# %%
