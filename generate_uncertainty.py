import torch
import numpy as np
import pinn
import matplotlib.pyplot as plt
import json

x = np.linspace(0, 1, 50)
t = np.linspace(0, 1, 50)

X, T = np.meshgrid(x, t)
# every possible combination of x and t from the arrays above (2D coordinates)
X_flat = X.ravel()
# flattened out meshgrid with all x values in an array and teh same for t below
T_flat = T.ravel()

X_ten = torch.tensor(X_flat, dtype=torch.float32, requires_grad=True).reshape(2500, 1)
T_ten = torch.tensor(T_flat, dtype=torch.float32, requires_grad=True).reshape(2500, 1)
# Converting the array to tensor of a specific type - float32 and reshaping 2500, 1
# The original shape was 2500, 
# required_grad=True --> remembers the gradient calculations of those tensor

model = pinn.PINN()
model.load_state_dict(torch.load("models/heat_pinn.pth"))

mean, std = pinn.mc_dropout_predict(model, X_ten, T_ten, 200)
# pulled from pinn.py

mean_grid = mean.detach().numpy().reshape(50, 50)
# We are reshaping the tensor to a grid
# (2500, 1) to 50x50
# it tells us teh mean temperature over each spot for the entire domain over 200 passes
# its the u(x,t) at each grid point
std_grid = std.detach().numpy().reshape(50, 50)
std_mean = np.mean(std_grid)
# this calculates the average uncertainty across the domain (a 2D 50x50 array)

std_max = np.max(std_grid)
row, col = np.unravel_index(np.argmax(std_grid), std_grid.shape)
# std_grid.shape tells the function that this 2D array is 50x50, it is needed to backtrack from a flat position
# np.argmax(std_grid) scans the entire grid and returns the INDEX of the largest value in the 1D array (flattened version of the 2D array)
# argmax tells us the number in a 1D array - something like 157, useful but wrong format
# unravel_index is the translator that takes that huge number and returns the row and col to pin point the largest value in the 2D array

x_largest = x[col]
t_largest = t[row]
# row and col are normal numbers (whole numbers) and find the coordinates which are values of x and t - this is how we do it

du_dx = np.gradient(mean_grid, axis=1)
# this gives us an array for the slopes along the data points
# axis=1, tells np to differentiate with respect to this axis (wqe have two 0 and 1)
# so it differentiates across columns for each fixed row
# shape of du_dx is 50x50 same as mean_grid

grad_max = np.max(np.abs(du_dx))
# we are trying to find the steepest slope which means it doesnt matter if it decreaases or increases - hence we use teh absolute values of the array entries
# it will only return one number, but we keep the negative slopes in teh competition by taking their absolute value

row2, col2 = np.unravel_index(np.argmax(np.abs(du_dx)), du_dx.shape)

x_max_grad = x[col2]
t_max_grad = t[row2]
# these are the coordinates to the point where the gradient of temperature is maximum

uncertainty_max_tempgrad = std_grid[row2, col2]
# given the coordinates from above - we are finding the uncertainty at that point
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(std_grid, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='hot')
# The line above displays a 2D array as an image
# ax is an Axes object and imshow means image show
# cmap controls teh color
plt.colorbar(im, ax=ax, label='Uncertainty (std)')

contours = ax.contour(X, T, mean_grid, levels=10, colors='white', linewidths=0.8)
ax.clabel(contours, inline=True, fontsize=7)

ax.set_xlabel('x (position)')
ax.set_ylabel('t (time)')
ax.set_title('MC Dropout Uncertainty — Full (x,t) Domain')

plt.savefig('figures/fig_uncertainty.png', dpi=300, bbox_inches='tight')
# DPI: dots per inch, measures the spatial resolution and pixel density of the saved image

results_dict = {
    "Mean uncertainty across the domain": float(std_mean),
    "Max uncertainty across the domain": float(std_max),
    "Max uncertainty co-ordinates": {"x" : float(x_largest), "t" : float(t_largest)},
    "Max gradient of temperature": {"x" : float(x_max_grad), "t" : float(t_max_grad)},
    "Uncertainty at max gradient of temperature" : float(uncertainty_max_tempgrad)
}

with open("benchmarks/benchmark_phase2.json", "r") as f:
    data = json.load(f)
data.update(results_dict)
with open("benchmarks/benchmark_phase2.json", "w") as f:
    json.dump(data, f, indent=4)