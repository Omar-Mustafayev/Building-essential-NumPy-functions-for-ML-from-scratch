import numpy as np # type: ignore
from sklearn.datasets import fetch_california_housing # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import ElasticNet # type: ignore
np.set_printoptions(suppress=True)


housing = fetch_california_housing()
X = housing.data  
y_train = housing.target.reshape(-1, 1)  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled)) 


n = X_train.shape[0]
m = X_train.shape[1]
theta_init = np.random.rand(m, 1)

def compute_cost(X, y, theta, lambda_reg, l1_ratio):
    errors = (X @ theta - y)**2
    mse = np.sum(errors)
    cost = mse + lambda_reg * (
        l1_ratio * np.sum(abs(theta[1:])) + 
        (1-l1_ratio) * 0.5 * (np.sum(theta[1:]**2))
        )
    return cost / (2 * n)

def compute_gradient(X, y, theta, lambda_reg, l1_ratio):
    err = X @ theta - y
    grad_mse = (X.T @ err) / n
    grad_l1 = np.zeros_like(theta)
    grad_l2 = np.zeros_like(theta)
    grad_l1[1:] = (lambda_reg / n * l1_ratio) * np.sign(theta[1:])
    grad_l2[1:] = (lambda_reg / n * (1-l1_ratio)) * theta[1:]
    gradients = grad_mse + grad_l1 + grad_l2
    return gradients


def gradient_descent(X, y, theta_init, lambda_reg, l1_ratio, learning_rate, iterations):
    thetas = theta_init.copy()
    for iter in range(iterations + 1):
        if iter % 10 == 0:
            cost = compute_cost(X, y, thetas, lambda_reg, l1_ratio)
            print(f"Iteration: {iter}  Thetas: {thetas.ravel()}  Cost: {cost}")
        gradient = compute_gradient(X, y, thetas, lambda_reg, l1_ratio)
        thetas-= learning_rate * gradient
    return thetas


final_thetas = gradient_descent(X_train, y_train, theta_init, lambda_reg = 0.2, l1_ratio=0.1, learning_rate=0.01, iterations=10000)



el_net = ElasticNet(alpha=0.2, l1_ratio=0.1, fit_intercept=False)
el_net.fit(X_train, y_train)
print("Intercept:", el_net.intercept_)
print("Coefficients:", el_net.coef_)
print("Our intercept:", final_thetas[0])
print("Our coefficients:", final_thetas[1:].ravel())

y_pred_custom = X_train @ final_thetas
y_pred_elastic = el_net.predict(X_train).reshape(-1, 1)

from sklearn.metrics import mean_squared_error
print("MSE (custom):", mean_squared_error(y_train, y_pred_custom))
print("MSE (ElasticNet):", mean_squared_error(y_train, y_pred_elastic))