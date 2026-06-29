import torch
import numpy as np
import pinn
import matplotlib.pyplot as plt

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

mean_grid = mean.detach().numpy().reshape(50, 50)
# We are reshaping the tensor to a grid
# (2500, 1) to 50x50
std_grid = std.detach().numpy().reshape(50, 50)

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(std_grid, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='hot')
plt.colorbar(im, ax=ax, label='Uncertainty (std)')

contours = ax.contour(X, T, mean_grid, levels=10, colors='white', linewidths=0.8)
ax.clabel(contours, inline=True, fontsize=7)

ax.set_xlabel('x (position)')
ax.set_ylabel('t (time)')
ax.set_title('MC Dropout Uncertainty — Full (x,t) Domain')

plt.savefig('figures/fig_uncertainty.png', dpi=300, bbox_inches='tight')