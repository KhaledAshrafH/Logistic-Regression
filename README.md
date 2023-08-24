# Logistic Regression from Scratch in Python

This program implements logistic regression from scratch using the gradient descent algorithm in Python to predict whether customers will purchase a new car based on their age and salary.

## Dataset

The dataset used in this program is `customer_data.csv`, which contains 400 records representing some of the company’s previous customers. The customer data is composed of the customer’s age and salary. The final column (i.e., the target) is a boolean value (0 if the customer didn’t purchase a new car and 1 if he/she purchased a new car).

## Preprocessing

The program performs the following preprocessing steps on the dataset:

- Normalize the feature data (age and salary) using min-max scaling to bring them into the range [0, 1].
- Shuffle the dataset rows to avoid any bias or order effects.
- Split the dataset into training (80%) and testing (20%) sets.

## Logistic Regression

The program defines a class `LogisticRegression` that implements logistic regression from scratch using the gradient descent algorithm. The class has the following attributes and methods:

### `__init__(self, x_train, y_train, w, b, numOfTrainingItems, learning_rate)`

The constructor of the class that takes the following parameters:

- `x_train`: The feature matrix of the training set.
- `y_train`: The target vector of the training set.
- `w`: The initial weight vector for the hypothesis function.
- `b`: The initial bias term for the hypothesis function.
- `numOfTrainingItems`: The number of training examples.
- `learning_rate`: The learning rate for gradient descent.

### `Sigmoid_Hypothesis(self, w, x, b)`

A method that takes the following parameters:

- `w`: The weight vector for the hypothesis function.
- `x`: The feature matrix of the input data.
- `b`: The bias term for the hypothesis function.

Returns: The output vector of the hypothesis function, which is computed as: `h(x) = 1 / (1 + e^-(w^T*x + b))`

### `Cost_Function(self)`

A method that computes the cost function for logistic regression, which is given by: `J(w, b) = -(1/m) * sum(y*log(h(x)) + (1-y)*log(1-h(x)))`

Returns: The value of the cost function for the current values of `w` and `b`.

### `gradient_descent(self)`

A method that performs one iteration of gradient descent to update the values of `w` and `b` using the following formulas:
```
w := w + (α/m) * X^T * (y - h(X))
b := b + (α/m) * sum(y - h(x))
```

Returns: The updated values of `w` and `b`.

### `train(self)`

A method that trains the logistic regression model by running gradient descent for a fixed number of iterations (10000 in this case).

Returns: The final values of `w` and `b` after training.

### `predict(self, x_test)`

A method that takes the following parameter:

- `x_test`: The feature matrix of the test set.

Returns: A vector of predictions for the test set, where each prediction is either 0 or 1 depending on whether the output of the hypothesis function is greater than or equal to 0.5 or not.

### `calc_accuracy(self, y_test, y_predicted)`

A method that takes the following parameters:

- `y_test`: The target vector of the test set.
- `y_predicted`: The vector of predictions for the test set.

Returns: The accuracy of the model on the test set, which is computed as: `total number of predictions / number of correct predictions`

## Results

The program prints out the following results:

- The final values of `w` and `b` after training.
- The accuracy of the model on the test set.
- A plot of the decision boundary and the data points.

## Customization

The program allows the user to customize some of the parameters and settings of the logistic regression model. The user can change the following variables in the code:

- `learning_rate`: The learning rate for gradient descent. The default value is 0.01, but the user can try different values and see how this affects the error or accuracy of the model.
- `numOfIterations`: The number of iterations for gradient descent. The default value is 10000, but the user can increase or decrease this value and see how this affects the convergence and performance of the model.
- `test_size`: The proportion of the dataset to be used as the test set. The default value is 0.2, which means 20% of the data will be used for testing and 80% for training. The user can change this value and see how this affects the accuracy of the model.

## Limitations and Future Work

The program has some limitations and areas for improvement, such as:

- The program only uses two features (age and salary) to predict the target variable (purchased). There may be other features that are relevant for predicting customer behavior, such as gender, education, location, etc. The program could be extended to include more features and perform feature selection or dimensionality reduction techniques to find the optimal subset of features.

- The program assumes that the data is linearly separable, which means that there exists a straight line that can separate the two classes (purchased or not purchased). However, this may not be true for all datasets, and some data may require more complex decision boundaries. The program could be modified to use other types of logistic regression models, such as polynomial logistic regression or regularized logistic regression, to handle non-linear or overfitting cases.

- The program uses a fixed threshold of 0.5 to classify the output of the hypothesis function as either 0 or 1. However, this may not be optimal for all scenarios, and some applications may require different levels of sensitivity or specificity. The program could be improved to use a variable threshold or a ROC curve analysis to find the best trade-off between true positive and false positive rates.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


## Authors

- [Khaled Ashraf Hanafy Mahmoud - 20190186](https://github.com/KhaledAshrafH).
- [Noura Ashraf Abdelnaby Mansour - 20190592](https://github.com/NouraAshraff).

## License

This program is licensed under the [MIT License](LICENSE.md).




