# Machine Learning Specialization - Supervised Machine Learning: Regression and Classification

Certificate - [https://coursera.org/share/5cede29aa40ee121b00e0954fd9842dd](https://coursera.org/share/5cede29aa40ee121b00e0954fd9842dd)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled.png)

Course 1

# 1. ****Supervised Machine Learning: Regression and Classification****

## 1.1 What is Machine Learning

Science of getting computers to learn without being explicitly programmed.

## 1.2 Introduction

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%201.png)

### 1.2.1 Supervised Learning

- Supervised learning is a type of machine learning where the model is trained on labeled data.
- This means that the data used to train the model includes both input data and corresponding correct output labels.
- The goal of supervised learning is to build a model that can make predictions about new, unseen examples by generalizing from the relationships learned in the training data.
- For example, a supervised learning algorithm for spam detection might be trained on a dataset of emails that are labeled as either "spam" or "not spam."
- The algorithm would learn to predict whether a new email is spam or not by identifying patterns in the features of the email that are indicative of spam.
- Some common applications of supervised learning include image classification, speech recognition, and natural language processing.

### 1.2.2 Unsupervised Learning

- Unsupervised learning is a type of machine learning where the model is not given any labeled training data.
- The goal of unsupervised learning is to learn the underlying structure of the data by identifying patterns and relationships within the data. Some common applications of unsupervised learning include clustering, anomaly detection, and data compression.
- Unsupervised learning is more difficult than supervised learning, as the model must learn to identify patterns in the data without the help of correct labels.

### 1.2.3 ✅ Practice quiz : Supervised vs unsupervised learning

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%202.png)

## 1.3 Regression Model

### 1.3.1 Linear Regression Model

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%203.png)

- Linear regression is a statistical method used to model the linear relationship between a dependent variable and one or more independent variables.
- It is a supervised learning algorithm that is used to make predictions about a continuous output variable based on the linear relationship between the input data and the output variable.
- Linear regression is often used to make predictions about continuous variables, such as predicting the price of a house based on its size, or predicting the likelihood of a customer making a purchase based on their income.
- It can also be used to test whether there is a significant relationship between two variables.

![Terminologies used in Machine Learning](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%204.png)

Terminologies used in Machine Learning

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%205.png)

- The function can be called as hypothesis also.

### 1.3.2 🔬 Optional Lab : Model Representation

[Machine-Learning-Specialization-Coursera/C1_W1_Lab03_Model_Representation_Soln.ipynb at main · rusirugunaratne/Machine-Learning-Specialization-Coursera](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week1/Optional%20Labs/C1_W1_Lab03_Model_Representation_Soln.ipynb)

### 1.3.3 Cost function formula

- In machine learning, a cost function is a measure of how well a model is able to predict the expected output.
- It is a measure of the **difference between the predicted output and the actual output**, and the goal of training a machine learning model is to **minimize** the cost function.
- The cost function is used to optimize the model's parameters, which are the values that determine how the model makes its predictions.
- The model's parameters are adjusted to minimize the cost function during training, so that the model makes better predictions on new data.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%206.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%207.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%208.png)

### 1.3.4 Cost function intuition

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%209.png)

How the cost function values changes when W changes. To make things simple we are setting the b as 0.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2010.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2011.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2012.png)

By plotting J(w) for different w values the following graph can be obtained.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2013.png)

By choosing a w value that minimize J(w) will give a better model that fits the data.

### 1.3.5 Cost function visualization

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2014.png)

Now we are considering both w and b. The J plot will look like the following.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2015.png)

To make this a 2D plot we can use a plot called the contour plot.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2016.png)

The above plot shows that the contour plot shows the areas with same values using lines and colors.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2017.png)

Hence we can use a contour plot to find the minimum J and corresponding w and b.

### 1.3.6 Visualization Examples

High cost

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2018.png)

Lower cost

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2019.png)

Minimum cost : the line ***perfectly*** fits the data

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2020.png)

### 1.3.7 🔬 Optional Lab : Cost function

[Machine-Learning-Specialization-Coursera/C1_W1_Lab04_Cost_function_Soln.ipynb at main · rusirugunaratne/Machine-Learning-Specialization-Coursera](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week1/Optional%20Labs/C1_W1_Lab04_Cost_function_Soln.ipynb)

### 1.3.8 ✅ Practice quiz : Regression model

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2021.png)

## 1.4 Train the model with gradient descent

### 1.4.1 Gradient descent

- Gradient Descent is known as one of the most commonly used optimization algorithms to train machine learning models by means of minimizing errors between actual and expected results.
- Further, gradient descent is also used to train Neural Networks.
- In mathematical terminology, Optimization algorithm refers to the task of minimizing/maximizing an objective function f(x) parameterized by x.
- Similarly, in machine learning, optimization is the task of minimizing the cost function parameterized by the model's parameters.
- The main objective of gradient descent is to minimize the convex function using iteration of parameter updates.
- Once these machine learning models are optimized, these models can be used as powerful tools for Artificial Intelligence and various computer science applications.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2022.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2023.png)

- The minimum J we get differs according to the initial w and b that we selected. These are called local minima.
- A local minimum of a function is a point at which the function has a lower value than at any nearby points. In the context of gradient descent, a local minimum refers to a point at which the cost function has a lower value than at any nearby points on the cost function's surface.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2024.png)

### 1.4.2 Implementing gradient descent

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2025.png)

- The two equations updates the values of w and b in a smaller amount simultaneously. The process is repeated until J reaches a local minima.
- The learning rate in gradient descent is a hyperparameter that determines the step size at which the algorithm updates the parameters of the model. It controls how fast or slow the model learns from the data. A smaller learning rate means that the model takes smaller steps towards the minimum of the loss function, and as a result, the training process is slower.
- A larger learning rate means that the model takes larger steps towards the minimum of the loss function, and as a result, the training process is faster. It is important to choose an appropriate learning rate because if the learning rate is too large, the model may overshoot the minimum of the loss function and may not converge.
- If the learning rate is too small, the model may take a long time to converge.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2026.png)

### 1.4.3 Gradient descent intuition

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2027.png)

### 1.4.4 Learning rate

Learning rate should be selected appropriately. Otherwise the the efficiency of the algorithm will decrease.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2028.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2029.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2030.png)

### 1.4.5 Gradient descent for linear regression

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2031.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2032.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2033.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2034.png)

### 1.4.6 Running gradient descent

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2035.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2036.png)

### 1.4.7 🔬 Optional lab : Gradient descent

[Machine-Learning-Specialization-Coursera/C1_W1_Lab05_Gradient_Descent_Soln.ipynb at main · greyhatguy007/Machine-Learning-Specialization-Coursera](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week1/Optional%20Labs/C1_W1_Lab05_Gradient_Descent_Soln.ipynb)

### 1.4.8 ✅ Practice quiz : Train the model with gradient descent

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2037.png)

---

## 1.5 Multiple Linear Regression

### 1.5.1 Multiple Features

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2038.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2039.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2040.png)

### 1.5.2 Vectorization

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2041.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2042.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2043.png)

### 1.5.3 🔬 **Optional Lab: Python, NumPy and Vectorization**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning%3A Regression and Classification/week2/Optional Labs/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week2/Optional%20Labs/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb)

### 1.5.4 **Gradient descent for multiple linear regression**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2044.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2045.png)

### 1.5.5 🔬 **Optional Lab: Multiple linear regression**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning%3A Regression and Classification/week2/Optional Labs/C1_W2_Lab02_Multiple_Variable_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week2/Optional%20Labs/C1_W2_Lab02_Multiple_Variable_Soln.ipynb)

## 1.6 Gradient Descent in Practice

### 1.6.1 Feature Scaling

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2046.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2047.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2048.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2049.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2050.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2051.png)

### 1.6.2 **Checking gradient descent for convergence**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2052.png)

### 1.6.3 Choosing a learning rate

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2053.png)

When choosing a learning rate, try with different learning rates and try to find the best learning rate.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2054.png)

### 1.6.4 🔬 Optional Lab: Feature Scaling

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning%3A Regression and Classification/week2/Optional Labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week2/Optional%20Labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb)

### 1.6.5 Feature Engineering

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2055.png)

### 1.6.6 Polynomial Regression

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2056.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2057.png)

### 1.6.7 🔬 Optional Lab: **Optional lab: Feature engineering and Polynomial regression**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning%3A Regression and Classification/week2/Optional Labs/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week2/Optional%20Labs/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb)

### 1.6.8 🔬 **Optional lab: Linear regression with scikit-learn**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning%3A Regression and Classification/week2/Optional Labs/C1_W2_Lab05_Sklearn_GD_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week2/Optional%20Labs/C1_W2_Lab05_Sklearn_GD_Soln.ipynb)

### 1.6.9 🔬 Practical lab: Linear regression

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning%3A Regression and Classification/week2/C1W2A1/C1_W2_Linear_Regression.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/week2/C1W2A1/C1_W2_Linear_Regression.ipynb)

## 1.7 Classification with logistic regression

### 1.7.1 Motivation

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2058.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2059.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2060.png)

### 1.7.2 🔬 Optional lab: Classification

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab01_Classification_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab01_Classification_Soln.ipynb)

### 1.7.3 Logistic Regression

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2061.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2062.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2063.png)

### 1.7.4 🔬 **Optional lab: Sigmoid function and logistic regression**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab02_Sigmoid_function_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab02_Sigmoid_function_Soln.ipynb)

### 1.7.5 Decision Boundary

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2064.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2065.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2066.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2067.png)

### 1.7.6 🔬 **Optional lab: Decision boundary**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab03_Decision_Boundary_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab03_Decision_Boundary_Soln.ipynb)

### 1.7.7 Cost function for logistic regression

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2068.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2069.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2070.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2071.png)

### 1.7.8 🔬 **Optional lab: Logistic loss**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab04_LogisticLoss_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab04_LogisticLoss_Soln.ipynb)

### 1.7.9 Simplified cost function for logistic regression

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2072.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2073.png)

### 1.7.10 🔬 **Optional lab: Cost function for logistic regression**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab05_Cost_Function_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab05_Cost_Function_Soln.ipynb)

### 1.7.11 **Gradient Descent Implementation**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2074.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2075.png)

### 1.7.12 🔬 **Optional lab: Logistic regression with scikit-learn**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab07_Scikit_Learn_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab07_Scikit_Learn_Soln.ipynb)

## 1.8 Overfitting

### 1.8.1 **The problem of overfitting**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2076.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2077.png)

### 1.8.2 **Addressing overfitting**

The following are the options we have to address this issue

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2078.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2079.png)

⬆️ We can choose the features that are necessary by intuition. But this can result in loosing some valuable information which could be used in the model.

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2080.png)

Summary

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2081.png)

### 1.8.3 🔬 **Optional lab: Overfitting**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab08_Overfitting_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab08_Overfitting_Soln.ipynb)

### 1.8.4 **Cost function with regularization**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2082.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2083.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2084.png)

### 1.8.5 **Regularized linear regression**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2085.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2086.png)

### 1.8.6 **Regularized logistic regression**

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2087.png)

![Untitled](Machine%20Learning%20Specialization%20-%20Supervised%20Machi%201dcfa5ef1a3247858c00182a5e0a9ffa/Untitled%2088.png)

### 1.8.7 🔬 **Optional lab: Regularization**

[https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1 - Supervised Machine Learning: Regression and Classification/week3/Optional Labs/C1_W3_Lab09_Regularization_Soln.ipynb](https://github.com/rusirugunaratne/Machine-Learning-Specialization-Coursera/blob/main/C1%20-%20Supervised%20Machine%20Learning:%20Regression%20and%20Classification/week3/Optional%20Labs/C1_W3_Lab09_Regularization_Soln.ipynb)