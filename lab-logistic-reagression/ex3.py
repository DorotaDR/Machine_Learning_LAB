# using real data, optimize classifier to predict given values

# split dataset into a training set and a test set
# train model on the training set
# calculate TP, FP, TN, FN on test set
# calculate sensitivity, specificity, positive predictivity and negative predictivity


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Classifier:
    theta = None

    def fit(self, X, y, nb_epochs=10000, eps=0.00001, lr=0.2):
        if X.shape[1] != len(y):
            X = X.T

        X_extented = self._extend_X(X)
        theta = np.zeros((X_extented.shape[0], 1))
        last_cost = 99

        for i in range(nb_epochs):

            h = 1 / (1 + np.exp(-theta.T @ X_extented))
            crossentropy = -y * np.log(h + eps) - (1 - y) * np.log(1 - h + eps)
            [cost] = np.sum(crossentropy, axis=1) / len(y)

            theta_deriv = sum((h - y) @ X_extented.T) / len(y)
            theta_deriv.shape = [len(theta_deriv), 1]

            # for j in range(len(theta)):
            #     theta[j] = theta[j] - lr * theta_deriv[j]
            theta = theta - lr * theta_deriv

            print("iteration: ", str(i + 1), ", cost: ", cost)

            if np.abs(last_cost - cost) < eps:
                break

            last_cost = cost

        self.theta = theta
        print(theta)

    def predict(self, X):
        if self._is_fitted():

            eps = 0.0001
            X_extented = self._extend_X(X)

            h = 1 / (1 + np.exp(-self.theta.T @ X_extented))

            # pos_pred = np.where(h>= 0.5 - eps)
            neg_pred = np.where(h < 0.5 - eps)

            y_pred = np.ones((1, X.shape[1]))
            y_pred[neg_pred] = 0

            return y_pred
        else:
            print("Please fit the classifier before use.")
            return -1

    def _is_fitted(self):
        return not (self.theta is None)

    def _extend_X(self, X):
        X_extented = np.ones((X.shape[0] + 1, X.shape[1]))
        for i in range(X.shape[0]):
            X_extented[i + 1, :] = X[i]

        return X_extented

    def plot(self, X):

        neg_pred = np.where(y == 0)[0]
        pos_pred = np.where(y == 1)[0]

        index_1 = 0
        index_2 = 1

        x_samples = np.ones((X.shape[0] + 1, 100))
        x1_tmp = np.linspace(min(X[index_1, :]), max(X[index_1, :]), 100)
        x2_tmp = np.linspace(min(X[index_2, :]), max(X[index_2, :]), 100)
        x_samples[index_1 + 1, :] = x1_tmp
        x_samples[index_2 + 1, :] = x2_tmp

        # decision_b_samples = -self.theta.T @ x_samples

        # TODO Only for 2-dim:
        decision_b_samples = (-x1_tmp * self.theta[1] - self.theta[0]) / self.theta[2]

        plt.plot(X[index_1, pos_pred], X[index_2, pos_pred], 'o')
        plt.plot(X[index_1, neg_pred], X[index_2, neg_pred], 'x')
        plt.plot(x1_tmp, decision_b_samples, '-')

        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.show()


data = pd.read_csv('./data.txt')
data = data.values

X = data[:, [0, 1]]
y = data[:, 2]
# y = [y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(X_train)

classifier = Classifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test.T)

print(y_pred)
print(y_test)

classifier.plot(X.T)
