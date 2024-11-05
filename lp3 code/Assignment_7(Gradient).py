import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative 
def function(x):
    return (x + 3) ** 2

def derivative(x): 
    return 2 * (x + 3)

# Gradient Descent Algorithm
def gradient_descent(starting_point, learning_rate, num_iterations): 
    x = starting_point
    history = [] # To store the values for plotting

    for _ in range(num_iterations):
        x -= learning_rate * derivative(x) 
        history.append(x)
    return x, history

# Parameters 
starting_point = 2
learning_rate = 0.1
num_iterations = 100

# Run Gradient Descent
minima, history = gradient_descent(starting_point, learning_rate, num_iterations)

# Output results
print(f"Local minima found at x = {minima:.4f}, y = {function(minima):.4f}")

# Plotting the function and the descent path 
x_values = np.linspace(-6, 2, 100) 
y_values = function(x_values)

plt.plot(x_values, y_values, label='y = (x + 3)²')
plt.scatter(history, function(np.array(history)), color='red', label='Descent Path') 
plt.title('Gradient Descent to Find Local Minima')
plt.xlabel('x')
plt.ylabel('y') 
plt.legend() 
plt.grid() 
plt.show()
