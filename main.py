import numpy as np
import pandas as pd


class LogisticRegression:
    # Z = X*w+p => h(x)
    def __init__(self, x_train, y_train, w, b, numOfTrainingItems, learning_rate):
        self.x_train = x_train
        self.y_train = y_train
        self.w = w
        self.b = b
        self.numOfTrainingItems = numOfTrainingItems
        self.learning_rate = learning_rate

    # Sigmoid Function (Z = w*X+p) .. w=theta1 , p=theta0
    def Sigmoid_Hypothesis(self, w, x, b):
        z = np.dot(x, w) + b
        return 1 / (1 + np.e ** -z)

    # Cost Function
    def Cost_Function(self):
        hypothesis = self.Sigmoid_Hypothesis(self.w, self.x_train, self.b)
        return np.dot((self.x_train.T, self.y_train - hypothesis))

    # get the values of w,b by gradient descent
    def gradient_descent(self):
        hypothesis = self.Sigmoid_Hypothesis(self.w, self.x_train, self.b)
        self.w += (self.learning_rate * (np.dot(self.x_train.T, self.y_train - hypothesis)) / numOfTrainingItems)
        self.b += (self.learning_rate * (np.sum(self.y_train - hypothesis)) / numOfTrainingItems)
        return [self.w, self.b]

    # training the data
    def train(self):
        for i in range(10000):
            self.w, self.b = self.gradient_descent()
        return [self.w, self.b]

    # predict new values
    def predict(self, x_test):
        hypothesis = self.Sigmoid_Hypothesis(self.w, x_test, self.b)
        return [1 if val >= 0.5 else 0 for val in hypothesis]

    def calc_accuracy(self, y_test, y_predicted):
        cnt = 0
        for i in range(len(y_test)):
            if y_test[i] == y_predicted[i]:
                cnt += 1
        return cnt / len(y_test)


# random initial values for w and b
w = np.random.random((2, 1))
b = np.random.random()

# loading the data (req1)
customerData = pd.read_csv('customer_data.csv')

# shuffle the data
customerData.sample(frac=1)
# print(customerData)


# X => values of features 1,2 (age, salary) & Y => values of output (purchased)
X = customerData.iloc[:, 0:2].values
Y = customerData.iloc[:, 2:3].values

# Feature Scaling using minmax normalization
X0_temp = []
X1_temp = []
for i in X:
    X0_temp.append(i[0])
    X1_temp.append(i[1])
minVal = min(X0_temp)
maxVal = max(X0_temp)
for i in range(len(X0_temp)):
    X0_temp[i] = (X0_temp[i] - minVal) / (maxVal - minVal)

minVal = min(X1_temp)
maxVal = max(X1_temp)
for i in range(len(X1_temp)):
    X1_temp[i] = (X1_temp[i] - minVal) / (maxVal - minVal)

X = X.astype(float)
for i in range(len(X)):
    X[i][0] = (X0_temp[i])
    X[i][1] = (X1_temp[i])

# Split the dataset into training and testing sets (req2)
X_Train = X[0:320]
Y_Train = Y[0:320]
X_Test = X[320:]
Y_Test = Y[320:]
numOfTrainingItems = int(X_Train.size / 2)
# print(numOfTrainingItems)


# logistic regression (req3)
LogisticReg = LogisticRegression(X_Train, Y_Train, w, b, numOfTrainingItems, 0.2)
w, b = LogisticReg.train()

# predictions on new data (req4)
Y_Predict = LogisticReg.predict(X_Test)
print(f"{Y_Predict}")

# Calculate the accuracy (req5)
print(f"Accuracy : {LogisticReg.calc_accuracy(Y_Test, Y_Predict) * 100}%")
