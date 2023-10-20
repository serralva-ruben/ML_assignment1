import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

group = 10
start = (group-1)*20+2
end = group*20+1


#reading the data
data = pd.read_excel('./data.xlsx',skiprows=start-1,nrows=(end-start), names=['x','y'])

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
alpha = 1
num_iterations = 4

# Keep a history of costs for plotting later
J_history = []

# Setup the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(normalized_data['x'], normalized_data['y'], label='Data Points')
line, = ax.plot(normalized_data['x'], np.zeros_like(normalized_data['y']), color='red', label='Regression Line')

ax.set_title('Scatter plot of x versus y with Gradient Descent in Real-time')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.legend()

# Update function for animation
def update(i):
    global theta
    theta, cost = single_step_gradient_descent(X_b, y, theta, alpha)
    J_history.append(cost)  # Store the cost for this iteration
    line.set_ydata(X_b.dot(theta))
    return line,

# Animate
ani = FuncAnimation(fig, update, frames=num_iterations, repeat=False, interval=1)

plt.show()

# You can still plot the cost function over iterations as well
plt.figure(figsize=(10, 6))
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function over iterations')
plt.show()