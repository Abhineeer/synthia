import numpy as np
import torch
import pinn
from solvers import heat_fd
import matplotlib.pyplot as plt
import time
import json

x_fd = np.linspace(0, 1, 100)
x_tensor = torch.reshape(torch.from_numpy(x_fd), (100, 1)).float()

t_point1 = torch.full((100,1), 0.1).float()
t_point5 = torch.full((100,1), 0.5).float()
t_1 = torch.full((100,1), 1).float()
model = pinn.PINN()
model.load_state_dict(torch.load("models/heat_pinn.pth"))

start_time_fd = time.perf_counter()
for i in range(10):
    x_out, t_out, u_out = heat_fd.solve_heat_fd(0.1, 100, 2500)
end_time_fd = time.perf_counter()
elapsed_time_fd = end_time_fd - start_time_fd
average_time_fd = elapsed_time_fd/10
# Average run time of FD (FD is a meaningfully longer process - 2500 steps compared to a singular PINN forward pass)

# u_out is the FD solver temperature output

model.eval()
# switches the network from training mode to evaluation (inference) mode
# in training mode the model changes weights and biases but in eval mode it doesnt, it works with the saved weights and biases

start_time_PINN = time.perf_counter()
for i in range(100):
    model(x_tensor, t_point1)
end_time_PINN = time.perf_counter()

elapsed_time_PINN = end_time_PINN - start_time_PINN
average_time_PINN = elapsed_time_PINN/100
# Timing the PINN run

speedup_ratio = average_time_fd/average_time_PINN
print(speedup_ratio)

# model.eval() switches dropout off while model.train() has it on. We need it on in the former to get different predictions each forward pass

prediction1 = model(x_tensor, t_point1)
prediction2 = model(x_tensor, t_point5)
prediction3 = model(x_tensor, t_1)
# PINN prediction for temperature
# prediction is basically a forward pass of model given the two inputs
# prediction is a (100, 1) tensor of temperature at 100 positions at a specific times (0.1, 0.5 and 1)



# COMPARING - PINN to FD and exact at t = 0.1, 0.5, and 1
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

u_exact = heat_fd.u_exact(x_fd, 0.1)

rel_pinn_exact_point1 = np.sqrt(np.mean((u_pinn - u_exact)**2))/np.sqrt(np.mean(u_exact**2))
print(rel_pinn_exact_point1)


plt.legend(["FD", "PINN"])
# plt.savefig("figures/fd_pinn_comparision_tpoint1.png")
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

u_exact = heat_fd.u_exact(x_fd, 0.5)

rel_pinn_exact_point5 = np.sqrt(np.mean((u_pinn - u_exact)**2))/np.sqrt(np.mean(u_exact**2))
print(rel_pinn_exact_point5)


plt.legend(["FD", "PINN"])
# plt.savefig("figures/fd_pinn_comparision_tpoint5.png")
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

u_exact = heat_fd.u_exact(x_fd, 1)

rel_pinn_exact_one = np.sqrt(np.mean((u_pinn - u_exact)**2))/np.sqrt(np.mean(u_exact**2))
print(rel_pinn_exact_one)


plt.legend(["FD", "PINN"])
# plt.savefig("figures/fd_pinn_comparision_t1.png")
plt.show()

results_dict = {
    "rel_l2_vs_fd": { "t_0.1": float(rel_l2_point1), "t_0.5": float(rel_l2_point5), "t_1.0": float(rel_l2_1)},
    "rel_l2_vs_exact": { "t_0.1": float(rel_pinn_exact_point1), "t_0.5": float(rel_pinn_exact_point5), "t_1.0": float(rel_pinn_exact_one)},
    "speedup_pinn_vs_fd": speedup_ratio,
    "training_time_seconds": 124.19095739995828
}

with open("benchmarks/benchmark_phase1.json", "w") as f:
    json.dump(results_dict, f, indent=4)

# NOTE
# the simulation runs from 0 to 1. 0.1, 0.5, and 1 are not seconds they represent how far along teh simulation are we
# 1 represetns end of simulation
