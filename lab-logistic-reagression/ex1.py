# fit the sigmoid curve and calculate decision boundary using given dataset

# a cheat sheet:
# in an optimization loop
# first calculate hypothesis for each datapoint x in X: h = 1 / (1 + exp(-theta0-theta1*x))
# then calculate crossentropy: -y*log(h) - (1-y)*log(1-h)
# and cost: sum(crossentropy) / len(x)
# next calculate derivatives for theta 0 and theta1 (similar to those in linear regression)
# theta0_deriv = sum(h - y) / len(y), theta1_deriv = sum((h-y)*X)
# and then update tbheta weights
# theta = theta - lr*theta_deriv

# check if cost is getting lower through iterations
# if not, try to modify the learning rate

# calculating decision boundary might look like this:
# theta[0] + theta[1]*x = 0
# theta[1]*x = -theta[0]
# x = -theta[0]/theta[1]

# the result might look like below

from matplotlib import pyplot as plt
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 25], dtype=np.float32)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1], dtype=np.float32)

theta = np.array([0, 0], dtype=np.float32)

# optimization loop
iterations = 10000
min_cost = 100
eps = 0.00001
lr = 0.1
theta_deriv = [0, 0]
for i in range(iterations):

    h = 1/(1 + np.exp(-theta[0] - theta[1] * X))
    crossentropy = -y * np.log(h + eps) - (1 - y) * np.log(1 - h + eps)
    cost = sum(crossentropy) / len(X)

    theta_deriv[0] = sum(h - y) / len(y)
    theta_deriv[1] = sum((h - y) * X) / len(y)

    for j in range(len(theta)):
        theta[j] = theta[j] - lr * theta_deriv[j]

    print("iteration: ", str(i + 1), ", cost: ", cost)

    if np.abs(min_cost - cost) < eps:
        break

    min_cost = cost

print(theta)

x_linspace = np.linspace(min(X), max(X), 100)
y_linspace = 1 / (1 + np.exp(-theta[0] - theta[1]*x_linspace))

plt.plot(X, y, 'x')
plt.plot(x_linspace, y_linspace, '-')
border_indexes = np.where(y_linspace >=0.5-eps)
# print(border_indexes)
plt.axvline(x=x_linspace[border_indexes[0][0]])
plt.show()

