import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

group = 10
start = (group-1)*20+2
# Parameters
alpha = 1.34
num_iterations = 20

#reading the data
data = pd.read_excel('./data.xlsx',skiprows=start-1, nrows=20, names=['x','y'])
print(data)
#normalizing the data - linear transformation also called min max transformation used because its the most versatile method of normalization
def min_max_normalize(dataset):
    normalized_dataset = dataset.copy()
    for i in range(len(normalized_dataset)):
        normalized_dataset.at[i, 'x'] = (dataset['x'][i] - dataset['x'].min()) / (dataset['x'].max() - dataset['x'].min())
        normalized_dataset.at[i, 'y'] = (dataset['y'][i] - dataset['y'].min()) / (dataset['y'].max() - dataset['y'].min())
    return normalized_dataset

def get_x_b(dataset):
    return np.c_[np.ones((dataset['x'].shape[0], 1)), dataset['x']]

def get_theta(dataset):
    xb = get_x_b(dataset)
    return np.linalg.pinv(xb.T.dot(xb)).dot(xb.T).dot(dataset['y'])

# returns the regression line using least squares closed form
def get_reg_line(dataset):
    x_b = get_x_b(dataset)
    theta = get_theta(dataset)
    return x_b.dot(theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1/(2*m)) * np.sum(errors**2)

def single_step_gradient_descent(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradients = (1/m) * X.T.dot(errors)
    theta -= alpha * gradients
    cost = compute_cost(X, y, theta)
    return theta, cost

def plot_reg_line(X, theta):
    return X.dot(theta)

normalized_data = min_max_normalize(data)

X_b = get_x_b(normalized_data)
y = normalized_data['y'].values.reshape(-1, 1)

theta = np.zeros((2,1))

# Keep a history of costs for plotting later
J_history = []

# Setup the figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot for regression data
ax1.scatter(normalized_data['x'], normalized_data['y'], label='Data Points')
line, = ax1.plot(normalized_data['x'], np.zeros_like(normalized_data['y']), color='red', label='Regression Line')
ax1.set_title('Scatter plot of x versus y with Gradient Descent in Real-time')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()

# Initial setup for cost function plot
ax2.set_xlim(0, num_iterations)
ax2.set_ylim(0, compute_cost(X_b, y, theta) + 10)
ax2.set_ylabel('Cost')
ax2.set_title('Cost function over iterations')
cost_line, = ax2.plot([], [], label='Cost Function', color='blue')
ax2.legend()

# Update function for animation
def update(i):
    global theta
    theta, cost = single_step_gradient_descent(X_b, y, theta, alpha)
    J_history.append(cost)
    
    # Update regression line
    line.set_ydata(X_b.dot(theta))
    
    ax2.clear()
    ax2.plot(J_history, label='Cost Function')
    ax2.set_title('Cost function over iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cost')
    ax2.grid(True)
    ax2.legend()

    ax2.text(0.05, 0.95, f'Current Cost: {cost:.4f}', transform=ax2.transAxes, verticalalignment='top')

    return line

# Animate
ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False, interval=1)
plt.tight_layout()
plt.show()