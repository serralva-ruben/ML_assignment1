import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

group = 10
start = (group-1)*20+2
# Parameters
alpha = 1
num_iterations = 80

#reading the data
data = pd.read_excel('./data.xlsx',skiprows=start-1, nrows=20, names=['x','y'])

#normalizing the data - linear transformation also called min max transformation used because its the most versatile method of normalization
def min_max_normalize(dataset):
    normalized_dataset = dataset.copy()
    for i in range(len(normalized_dataset)):
        normalized_dataset.at[i, 'x'] = (dataset['x'][i] - dataset['x'].min()) / (dataset['x'].max() - dataset['x'].min())
        normalized_dataset.at[i, 'y'] = (dataset['y'][i] - dataset['y'].min()) / (dataset['y'].max() - dataset['y'].min())
    return normalized_dataset

# returns the regression line using least squares closed form
def compute_regression_line(dataset):
    x_b = np.c_[np.ones((dataset['x'].shape[0], 1)), dataset['x']]
    theta = np.linalg.pinv(x_b.T.dot(x_b)).dot(x_b.T).dot(dataset['y'])
    return x_b.dot(theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1/(2*m)) * np.sum(errors**2)

def single_step_gradient_descent(X, y, theta, alpha):
    print(f'θ1 : {theta[1]}\nθ0 : {theta[0]}')
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
print(normalized_data)
X_b = np.c_[np.ones((normalized_data['x'].shape[0], 1)), normalized_data['x']]
y = normalized_data['y'].values.reshape(-1, 1)
theta = np.random.rand(2,1)

# Keep a history of costs for plotting later
J_history = []

# Setup the figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Compute the regression line using least squares method
reg_line_least_squares = compute_regression_line(normalized_data)

# Scatter plot for regression data
ax1.scatter(normalized_data['x'], normalized_data['y'], label='Data Points')
ax1.plot(normalized_data['x'], reg_line_least_squares, color='green', label='Least Squares Regression Line')
line, = ax1.plot(normalized_data['x'], np.zeros_like(normalized_data['y']), color='red', label='Gradient Descent Regression Line')
line.set_ydata(X_b.dot(theta))
ax1.set_title('Linear Regression')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()

iteration_count = 0
# Update function for animation
def update(i):
    global iteration_count
    print(f'Current iteration: {iteration_count}')
    iteration_count += 1
    global theta
    theta, cost = single_step_gradient_descent(X_b, y, theta, alpha)
    J_history.append(cost)
    
    # Update regression line
    line.set_ydata(X_b.dot(theta))
    
    ax2.clear()
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost function for Gradient Descent Regression Line')
    ax2.grid(True)
    ax2.plot(J_history, label='Cost Function')
    ax2.legend()

    ax2.text(0.05, 0.95, f'Current Cost: {cost:.6f}', transform=ax2.transAxes, verticalalignment='top')

    return line

# Animate
ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False, interval=7500)
plt.show()