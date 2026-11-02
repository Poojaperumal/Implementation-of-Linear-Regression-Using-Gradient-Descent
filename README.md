# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start the program.

2. Import the required libraries such as NumPy and Pandas.

3. Load the dataset containing population and profit values of different cities.

4. Extract the independent variable (Population) and dependent variable (Profit).

5. Initialize the parameters (slope and intercept) with zero.

6. Set the learning rate and number of iterations.

7. Calculate the predicted profit using the linear regression equation.

8. Calculate the cost function (Mean Squared Error).

9. Update the slope and intercept using the gradient descent algorithm.

10. Repeat the process until the specified number of iterations is completed.

11. Predict the profit for the given city population.

12. Display the predicted profit.

13. Stop the program.

Result: Thus, the linear regression model using the gradient descent algorithm is implemented successfully to predict the profit of a city.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: POOJA P
RegisterNumber: 212224100041  
*/
```

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
```
```
data = pd.read_csv("exp_3_50_Startups.csv")
print("First 5 rows of the dataset:")
display(data.head())
```

<img width="602" height="245" alt="image" src="https://github.com/user-attachments/assets/c63c71d8-2346-4041-8b01-e91a53a0da90" />

```
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)
print("Encoded dataset:")
display(data_encoded.head())
```
<img width="732" height="270" alt="image" src="https://github.com/user-attachments/assets/95350e22-f7b6-439d-9093-967734f00646" />

```
X_raw = data_encoded.drop('Profit', axis=1).values
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)
```
<img width="306" height="62" alt="image" src="https://github.com/user-attachments/assets/42a17d75-f0cc-4c08-a8a6-0fc14f27ddab" />

```
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)
```
<img width="440" height="37" alt="image" src="https://github.com/user-attachments/assets/cb255214-ed53-4710-8c58-3ba465afe515" />

```
def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost

```

```
def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history

```
```
alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())
```

<img width="651" height="312" alt="image" src="https://github.com/user-attachments/assets/544bec23-a4ac-4c8b-8091-5d844744d0aa" />

```
plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()

```

<img width="732" height="455" alt="image" src="https://github.com/user-attachments/assets/8f1db9c7-5e50-4266-85a4-0d0132949e4a" />

```
new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

# Apply one-hot encoding
new_encoded = pd.get_dummies(new_sample, columns=['State'], drop_first=True)

# Align columns with training data (add missing ones as 0)
new_encoded = new_encoded.reindex(columns=data_encoded.drop('Profit', axis=1).columns, fill_value=0)

# Scale new data
new_scaled = X_scaler.transform(new_encoded)

# Add bias
new_design = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))

# Predict (scaled)
scaled_pred = new_design.dot(theta)

# Inverse transform to original units
pred_original = y_scaler.inverse_transform(scaled_pred)

print(f"\nPredicted Profit for the new startup: ₹{pred_original[0][0]:,.2f}")

```

<img width="766" height="82" alt="image" src="https://github.com/user-attachments/assets/837d3f51-3d7f-4afc-88d2-1e0b9e7df4af" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
