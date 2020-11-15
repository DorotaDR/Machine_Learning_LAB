# generalize optimization code for X being a matrix, where its rows are features and columns are examples
# code should work independently from number of features and number of examples
# use matrix multiplication (np.matmul or @)
# plot decision boundary on a plot x2(x1)
# calculating decision boundary might look like this:
# theta0 + theta1*x1 + theta2*x2 = 0
# theta2*x2 = -theta0 - theta1*x1
# x2 = -theta0/theta2 - theta1/theta2 * x1


from matplotlib import pyplot as plt
import numpy as np

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bias' 'variables' already appended to X
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 25],
              [13, 9, 8, 6, 4, 2, 1, 0, 3, 4, 2]], dtype=np.float32)

y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float32)

theta = np.zeros((X.shape[0], 1))

# optimization loop
iterations = 10000
min_cost = 100
eps = 0.00001
lr = 0.1

for i in range(iterations):

    h = 1 / (1 + np.exp(-theta.T @ X))
    crossentropy = -y * np.log(h + eps) - (1 - y) * np.log(1 - h + eps)
    [cost] = np.sum(crossentropy, axis=1) / X.shape[1]

    theta_deriv = sum((h - y) @ X.T) / X.shape[1]
    theta_deriv.shape = [len(theta_deriv), 1]

    # for j in range(len(theta)):
    #     theta[j] = theta[j] - lr * theta_deriv[j]
    theta = theta - lr * theta_deriv

    print("iteration: ", str(i + 1), ", cost: ", cost)

    if np.abs(min_cost - cost) < eps:
        break

    min_cost = cost

print(theta)

neg_pred = np.where(y == 0)[0]
pos_pred = np.where(y == 1)[0]

index_1 = 1
index_2 = 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

x_samples = np.ones((X.shape[0], 100))
x_tmp = np.linspace(min(X[index_1, :]), max(X[index_1, :]), 100)
x_samples[index_1, :] = x_tmp
y_samples = 1 / (1 + np.exp(-theta.T @ x_samples))

ax1.plot(X[index_1, pos_pred], y[pos_pred], 'o')
ax1.plot(X[index_1, neg_pred], y[neg_pred], 'x')
ax1.plot(x_tmp, y_samples[0, :], '-')
try:
    border_indexes = np.where(y_samples[0, :] >= 0.5 - eps)
    # print(border_indexes)
    ax1.axvline(x=x_samples[1, border_indexes[0][0]])
except IndexError as e:
    print(e)



x_samples = np.ones((X.shape[0], 100))
x1_tmp = np.linspace(min(X[index_1, :]), max(X[index_1, :]), 100)
x2_tmp = np.linspace(min(X[index_2, :]), max(X[index_2, :]), 100)
x_samples[index_1, :] = x1_tmp
x_samples[index_2, :] = x2_tmp

decision_b_samples = -theta.T @ x_samples

ax2.plot(X[index_1, pos_pred], X[index_2, pos_pred], 'o')
ax2.plot(X[index_1, neg_pred], X[index_2, neg_pred], 'x')
ax2.plot(x1_tmp, decision_b_samples[0, :], '-')

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.show()
