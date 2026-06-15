import numpy as np
import torch
import pinn
from solvers import heat_fd
import matplotlib.pyplot as plt


x_fd = np.linspace(0, 1, 100)
x_tensor = torch.reshape(torch.from_numpy(x_fd), (100, 1)).float()

t_point1 = torch.full((100,1), 0.1).float()
t_point5 = torch.full((100,1), 0.5).float()
t_1 = torch.full((100,1), 1).float()
model = pinn.PINN()
model.load_state_dict(torch.load("models/heat_pinn.pth"))

x_out, t_out, u_out = heat_fd.solve_heat_fd(0.1, 100, 2500)
# u_out is the FD solver temperature output

model.eval()

prediction1 = model(x_tensor, t_point1)
prediction2 = model(x_tensor, t_point5)
prediction3 = model(x_tensor, t_1)
# PINN prediction for temperature
# prediction is basically a forward pass of model given the two inputs
# prediction is a (100, 1) tensor of temperature at 100 positions at a specific times (0.1, 0.5 and 1)

# ------- t = 0.1s ------------
print(prediction1)

plt.plot(x_fd, u_out[int(0.1/t_out[1]), :])
plt.plot(x_tensor, prediction1.detach().numpy())

u_pinn = prediction1.detach().numpy().flatten()
# flatten converts (100,1) to (100,) which matches the shape for subtraction
u_fd_t01 = u_out[int(0.1/t_out[1]), :]

rel_l2_point1 = np.sqrt(np.mean((u_pinn - u_fd_t01)**2))/np.sqrt(np.mean(u_fd_t01**2))
# this is the relative L2 formula: ||| u_pinn - u_fd ||/ || u_fd ||
print(rel_l2_point1)

plt.legend(["FD", "PINN"])
plt.savefig("figures/fd_pinn_comparision_tpoint1.png")
plt.show()


# ------- t = 0.5s ------------
print(prediction2)

plt.plot(x_fd, u_out[int(0.5/t_out[1]), :])
plt.plot(x_tensor, prediction2.detach().numpy())

u_pinn = prediction2.detach().numpy().flatten()
# flatten converts (100,1) to (100,) which matches the shape for subtraction
u_fd_t05 = u_out[int(0.5/t_out[1]), :]

rel_l2_point5 = np.sqrt(np.mean((u_pinn - u_fd_t05)**2))/np.sqrt(np.mean(u_fd_t05**2))
# this is the relative L2 formula: ||| u_pinn - u_fd ||/ || u_fd ||
print(rel_l2_point5)

plt.legend(["FD", "PINN"])
plt.savefig("figures/fd_pinn_comparision_tpoint5.png")
plt.show()

# ------- t = 1s ------------
print(prediction3)

plt.plot(x_fd, u_out[int(1/t_out[1]), :])
plt.plot(x_tensor, prediction3.detach().numpy())

u_pinn = prediction3.detach().numpy().flatten()
# flatten converts (100,1) to (100,) which matches the shape for subtraction
u_fd_t1 = u_out[int(1/t_out[1]), :]

rel_l2_1 = np.sqrt(np.mean((u_pinn - u_fd_t1)**2))/np.sqrt(np.mean(u_fd_t1**2))
# this is the relative L2 formula: ||| u_pinn - u_fd ||/ || u_fd ||
print(rel_l2_1)

plt.legend(["FD", "PINN"])
plt.savefig("figures/fd_pinn_comparision_t1.png")
plt.show()
