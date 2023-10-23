import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

group = 10
start = (group-1)*20+2
end = group*20+1
# Parameters
alpha = 1
num_iterations = 300

# Reading the data
data = pd.read_excel('./data.xlsx', skiprows=start-1, nrows=(end-start), names=['x', 'y'])

# Normalizing the data - linear transformation also called min-max transformation 
def min_max_normalize(dataset):
    normalized_dataset = dataset.copy()
    for i in range(len(normalized_dataset)):
        normalized_dataset.at[i, 'x'] = (dataset['x'][i] - dataset['x'].min()) / (dataset['x'].max() - dataset['x'].min())
        normalized_dataset.at[i, 'y'] = (dataset['y'][i] - dataset['y'].min()) / (dataset['y'].max() - dataset['y'].min())
    return normalized_dataset

normalized_data = min_max_normalize(data)

# Design Matrix for Polynomial Regression
X = np.c_[normalized_data['x']**2, normalized_data['x'], np.ones(normalized_data.shape[0])]
y = normalized_data['y'].values.reshape(-1, 1)

# Initialize parameters theta
theta = np.zeros((3, 1))

# Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (4 * m)) * np.sum(errors**4)
    return cost

# Single Step Gradient Descent
def single_step_gradient_descent(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradients = (1 / m) * X.T.dot(errors**3)
    theta -= alpha * gradients
    cost = compute_cost(X, y, theta)
    return theta, cost

# Setup the figure and axis for the regression animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.scatter(normalized_data['x'], normalized_data['y'], label='Data Points')
x_vals = np.linspace(normalized_data['x'].min(), normalized_data['x'].max(), 400)
line, = ax1.plot(x_vals, np.zeros_like(x_vals), color='red', label='Regression Line')
ax1.set_title('Polynomial Regression using Gradient Descent')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()

J_history = []

# Update function for animation
def update(i):
    global theta
    theta, cost = single_step_gradient_descent(X, y, theta, alpha)
    J_history.append(cost)

    # Update regression line
    line.set_ydata(theta[0] * x_vals**2 + theta[1] * x_vals + theta[2])

    ax2.clear()
    ax2.plot(J_history, label='Cost Function')
    ax2.set_title('Cost function over iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cost')
    ax2.grid(True)
    ax2.legend()
    ax2.text(0.05, 0.95, f'Current Cost: {cost:.4f}', transform=ax2.transAxes, verticalalignment='top')

    return line

ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False, interval=1)
plt.tight_layout()
plt.show()
