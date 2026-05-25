# heat_fd - finite differences
# derived on the whiteboard

import numpy as np
import matplotlib.pyplot as plt


r = 0.4
alpha = 0.1

def u_exact(x, t):
        return np.sin(np.pi*x)*np.e**(-1*alpha*(np.pi)**2*t)

# x in u_exact is the spatial grid
# while it is x axis in plt.plot --> 2 jobs

def solve_heat_fd(alpha_new, n_x, n_t):

    x_new = np.linspace(0, 1, n_x)
    dx2 = x_new[1] - x_new[0]

    dt2 = r*(dx2)**2/alpha_new
    t2 = np.arange(0, (n_t + 1)*dt2, dt2) # we want n_t + 1 steps since we include t = 0
    # arange - (start, end, step) [end: number of steps * step]

    u_new = np.zeros((n_t + 1, n_x))
    u_new[0] = np.sin(np.pi*x_new)

    for n in range(n_t):
        u_new[n+1, 1:-1] = alpha_new*(dt2/dx2**2)*(u_new[n, 2:] + u_new[n, :-2] - 2*u_new[n, 1:-1]) + u_new[n, 1:-1]
    # numpy syntax --> manipulating arrays
    # 1:-1 mean interior points only --> they stay 0 forever
    # we are computing the next temperature given the current one


    return x_new, t2, u_new


    
x_out, t_out, u_out = solve_heat_fd(0.1, 100, 2500)
    

# ----- t = 0.1 -------
plt.plot(x_out, u_exact(x_out, 0.1))
plt.plot(x_out, u_out[245, :])
plt.legend(["Exact", "FD"])
# plt.savefig("figures/fd_validation_t01.png")
plt.show()

# ----- t = 0.5 -------
plt.plot(x_out, u_exact(x_out, 0.5))
plt.plot(x_out, u_out[int(0.5/t_out[1]), :])
plt.legend(["Exact", "FD"])
# plt.savefig("figures/fd_validation_t05.png")
plt.show()

# ----- t = 1 ---------
plt.plot(x_out, u_out[int(1/t_out[1]), :])
plt.plot(x_out, u_exact(x_out, 1))
plt.legend(["Exact", "FD"])
# plt.savefig("figures/fd_validation_t1.png")
plt.show()

# --------------------------------------------------------- Scratch Work -------------------------------------------------------------------

# N_x = 100 # spatial points
# N_t = 2500 # time steps -- updated after seeing that we run into a range issue trying to get the FD result for t = 0.1 s
# # given in the spec (SYNTHIA)

# x = np.linspace(0,1,100) # linspace is used to generate an evenly spaced array of numbers
# x is a spatial grid, every calculation needs to know where the points are
# think of the rod - you need to know all the positions on the rod to evaluate sin(pi*x)

# dx = x[1] - x[0] # spacing between the points in that grid - equal spacing dx is constant

# dt = r*(dx)**2/alpha # transforming stability parameter equation
# print(0.1/dt) = finding the FD row that corresponds to t = 0.1 s

# u = np.zeros((N_t + 1, N_x)) # array of shape N_t + 1 * N_x
# u is where we store the information (the entire solution) - every point every time step
# u is basically a spreadsheet
# every spatial point has a different temperature at different times - we store all of it

# u[0] = np.sin(np.pi*x) # first row changed to sin(pi*x)


# We set the initial condition (row 1) ==> u(x,0) = sin(pi*x), our initial condition (time step 0)
# At t = 0 the temperature at each point x is sin(pi*x)


