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
        #note only for 2-dim data

        X_extented = self._extend_X(X)

        neg_pred = np.where(y == 0)[0]
        pos_pred = np.where(y == 1)[0]

        index_1 = 1
        index_2 = 2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        x_samples = np.ones((X_extented.shape[0], 100))
        x_tmp = np.linspace(min(X_extented[index_1, :]), max(X_extented[index_1, :]), 100)
        x_samples[index_1, :] = x_tmp
        y_samples = 1 / (1 + np.exp(-self.theta.T @ x_samples))

        ax1.plot(X_extented[index_1, pos_pred], y[pos_pred], 'o')
        ax1.plot(X_extented[index_1, neg_pred], y[neg_pred], 'x')
        ax1.plot(x_tmp, y_samples[0, :], '-')
        try:
            border_indexes = np.where(y_samples[0, :] >= 0.5 - 0.0001)
            # print(border_indexes)
            ax1.axvline(x=x_samples[1, border_indexes[0][0]])
        except IndexError as e:
            print(e)

        x_samples = np.ones((X_extented.shape[0], 100))
        x1_tmp = np.linspace(min(X_extented[index_1, :]), max(X_extented[index_1, :]), 100)
        x2_tmp = np.linspace(min(X_extented[index_2, :]), max(X_extented[index_2, :]), 100)
        x_samples[index_1, :] = x1_tmp
        x_samples[index_2, :] = x2_tmp

        decision_b_samples = -self.theta[0] / self.theta[2] - self.theta[1] / self.theta[2] * x_samples

        ax2.plot(X_extented[index_1, pos_pred], X_extented[index_2, pos_pred], 'o')
        ax2.plot(X_extented[index_1, neg_pred], X_extented[index_2, neg_pred], 'x')
        ax2.plot(x1_tmp, decision_b_samples[1, :], '-')

        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.show()


    def calculate_TP_FP_TN_FN(self, X, y_true):
        #   result of the function could also be confusion matrix,
        #   but values returned as tuple will be more helpful in the excercise

        y_pred = self.predict(X)

        TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
        FP = np.sum(np.logical_and(y_pred == 0, y_true == 1))
        TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
        FN = np.sum(np.logical_and(y_pred == 1, y_true == 0))

        return (TP, FP, TN, FN)


data = pd.read_csv('./data.txt')
data = data.values

# Preprocess data

data[:, 0] = (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0])
data[:, 1] = (data[:, 1] - np.mean(data[:, 1])) / np.std(data[:, 1])

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

# calculate TP, FP, TN, FN on test set

(TP, FP, TN, FN) = classifier.calculate_TP_FP_TN_FN(X_test.T, y_test)

print(f"(TP={TP}, FP={FP}, TN={TN}, FN={FN})  ")

# calculate sensitivity, specificity, positive predictivity and negative predictivity

sensitivity = TP / (TP + FN)
positive_redictivity = TP / (TP + FP)

specificity = TN / (TN + FP)
negative_predictivity = TN / (TN + FP)

print(f"calculated sensitivity = {sensitivity},\n"
      f"specificity = {specificity}, \n"
      f"positive predictivity = {positive_redictivity} \n"
      f"and negative predictivity = {negative_predictivity}")

classifier.plot(X.T)
