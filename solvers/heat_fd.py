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


def compute_l2(u_out, u_exact):
     return np.sqrt(np.mean((u_out - u_exact)**2))
    
     
if __name__ == "__main__":    
    x_out, t_out, u_out = solve_heat_fd(0.1, 100, 2500)
        

    # ----- t = 0.1 -------
    plt.plot(x_out, u_exact(x_out, 0.1))
    plt.plot(x_out, u_out[245, :])
    print(compute_l2(u_out[245,:], u_exact(x_out, 0.1)))
    plt.legend(["Exact", "FD"])

    # plt.savefig("figures/fd_validation_t01.png")
    plt.show()

    # ----- t = 0.5 -------
    plt.plot(x_out, u_exact(x_out, 0.5))
    plt.plot(x_out, u_out[int(0.5/t_out[1]), :])
    print(compute_l2(u_out[int(0.5/t_out[1]),:], u_exact(x_out, 0.5)))
    plt.legend(["Exact", "FD"])
    # plt.savefig("figures/fd_validation_t05.png")
    plt.show()

    # ----- t = 1 ---------
    plt.plot(x_out, u_out[int(1/t_out[1]), :])
    plt.plot(x_out, u_exact(x_out, 1))
    print(compute_l2(u_out[int(1/t_out[1]),:], u_exact(x_out, 1)))
    plt.legend(["Exact", "FD"])
    # plt.savefig("figures/fd_validation_t1.png")
    plt.show()