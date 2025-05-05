import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # BP神经网络
from sklearn.ensemble import RandomForestRegressor  # 随机森林
from sklearn.svm import SVR  # 支持向量回归
from sklearn.metrics import mean_squared_error

# 加载实际数据集的函数
def load_actual_dataset(C:\Users\1\Desktop\ymx):
    """
    从CSV文件加载数据
    :param file_path: C:\Users\1\Desktop\ymx
    :return: 自变量 (X) 和 因变量 (Y)
    """
    # 读取CSV文件
    data = pd.read_csv(C:\Users\1\Desktop\ymx)
    
    # 假设CSV文件的前5列是自变量，后4列是因变量
    X = data.iloc[:, :5].values  # 取前5列作为自变量
    Y = data.iloc[:, 5:].values  # 取后4列作为因变量
    return X, Y

# BP神经网络
def train_bp_neural_network(X_train, Y_train, X_test, Y_test):
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print("BP神经网络 - 均方误差 (MSE):", mse)
    return model

# 随机森林
def train_random_forest(X_train, Y_train, X_test, Y_test):
    models = []
    mse_list = []
    for i in range(Y_train.shape[1]):  # 对每个因变量单独训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train[:, i])
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test[:, i], Y_pred)
        mse_list.append(mse)
        models.append(model)
    print("随机森林 - 每个因变量的均方误差 (MSE):", mse_list)
    return models

# 支持向量回归
def train_svr(X_train, Y_train, X_test, Y_test):
    models = []
    mse_list = []
    for i in range(Y_train.shape[1]):  # 对每个因变量单独训练模型
        model = SVR(kernel='rbf')
        model.fit(X_train, Y_train[:, i])
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test[:, i], Y_pred)
        mse_list.append(mse)
        models.append(model)
    print("支持向量回归 - 每个因变量的均方误差 (MSE):", mse_list)
    return models

# 主函数
if __name__ == "__main__":
    # 数据集文件路径
    file_path = "your_actual_dataset.csv"  # 替换为你的实际CSV文件路径

    # 加载实际数据集
    X, Y = load_actual_dataset(file_path)

    # 分割训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("开始训练模型...\n")
    
    # BP神经网络
    bp_model = train_bp_neural_network(X_train, Y_train, X_test, Y_test)
    
    # 随机森林
    rf_models = train_random_forest(X_train, Y_train, X_test, Y_test)
    
    # 支持向量回归
    svr_models = train_svr(X_train, Y_train, X_test, Y_test)
    
    print("\n模型训练完成！")
