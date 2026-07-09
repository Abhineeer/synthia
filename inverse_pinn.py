import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import RegularGridInterpolator
from solvers.heat_fd import solve_heat_fd
import matplotlib.pyplot as plt

class InPINN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InPINN, self).__init__(*args, **kwargs)

        self.alpha = nn.Parameter(data=torch.tensor([0.5]), requires_grad=True)
        # n.Parameter treats this tensor as a learnable weight
        # tensor 0.5 is because firstly that is the initial value subject to change and nn.Parameter only takes torch tensors

        
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, t):
        combined = torch.cat((x, t), dim=1)
        return self.network(combined)


    
def loss_pde(model):
    x = torch.rand(2000, 1, requires_grad=True)
    t = torch.rand(2000, 1, requires_grad=True)
    u = model(x,t)
    ut_firstpde = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)
    ux_firstpde = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    ux_secondpde = torch.autograd.grad(ux_firstpde[0], x, grad_outputs=torch.ones_like(ux_firstpde[0]), create_graph=True)
    alpha = model.alpha
    # calling the changing alpha, it is not preset

    L_pde = ((ut_firstpde[0] - alpha*ux_secondpde[0])**2).mean()

    # The general formula is same as before

    return L_pde

def loss_bc(model):
    x = torch.randint(low=0, high=2, size=(200,1)).float()
    t = torch.rand(200, 1)
    u = model(x, t)
    L_bc = (u**2).mean()

    return L_bc


def loss_ic(model):
    t = torch.randint(low=0, high=1, size=(200, 1)).float()
    x = torch.rand(200, 1)
    u_ic = torch.sin(torch.pi*x)

    u = model(x,t)
    loss_ic = ((u_ic - u)**2).mean()
    # I copied over the loss_bc and ic same as before no changes needed
    return loss_ic

x_in, t_in, u_in = solve_heat_fd(alpha_new=0.1, n_x=100, n_t=200)
T_max = t_in.max()

# in stands for data made for the interpolator
# We are using this clean data to train our interpolator - This is "clean" because we know its right from Phase 1 of SYNTHIA
# It comes from the Finite Difference solver in heat_fd.py
interpolator = RegularGridInterpolator((t_in, x_in), u_in)
# This marks the completion of the first step of using the internpolator
# t_in and x_in are the axes and u_in are the plot points

x_obs = torch.rand(250, 1)
t_obs = torch.rand(250, 1)*T_max
# ------ We went up form 150 to 250 random points to improve accuracy -------
# obs stands for observed - since this data is designed to be experimental
# torch.rand - 150 random points of x and t, [0, 1)
# Their shape is (150, 1) - rows,cols
combined_obs = torch.cat((t_obs, x_obs), dim=1)
# x and t in the same array of shape (150, 2)
#  2 columns of 150 values
combined_obs_array = combined_obs.numpy()
# converts the torch tensor to a numpy array, exactly the type we need for the interpolator

u_obs_array = interpolator(combined_obs_array)
# This is querying it, it will return 150 values of u [It is a numpy array]

noise = torch.randn(250,)*0.05
# This is what helps us simulate experimental data
# (150,) --> This is a 1D array, a flat sequence of numbers
# (150,1) --> This might look the same but its actually a 2D array. It has 1 column and 150 rows

u_obs = torch.from_numpy(u_obs_array)
# We convert it before adding the Gaussian noise because the noise is a tensor and adding a tensor to an array would throw an error, NO LIKEY
u_ex_data = u_obs + noise
u_ex_data = u_ex_data.reshape(-1, 1)
# We want this to be of shape (150, 1) specifially because loss_data needs it to be, by definition of the function

def loss_data(model, x_obs, t_obs, u_ex_data):
    u_pred = model(x_obs, t_obs)
    # loss_data constrains the u data, it pins them down. This causes the derivatives of u to be pinned down as well which in turn forces l_pde to use only alter alpha in order to gt the needed results for a fixed yet noisy u.

    L_data = ((u_ex_data - u_pred)**2).mean()
    return L_data

if __name__ == "__main__":
    torch.manual_seed(42)
    model = InPINN()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    alpha_vals = []
    step_count = []

    for i in range(50000):
        optimizer.zero_grad()
        loss1 = loss_pde(model)
        loss2 = loss_bc(model)
        loss3 = loss_ic(model) 
        loss4 = loss_data(model, x_obs, t_obs, u_ex_data)
        # Something to note - I thought having the same values for the arguments in loss4 was a problem, but not really thats the entire thign. We want to compare it to the same observed value.
        # Else it is simply a moving target, helps no one

        L_total = loss1 + loss2 + loss3 + loss4
        L_total.backward()

        optimizer.step()
        if i % 5000 == 0:
            print(model.alpha.item())
            print("---------------------")
            print(L_total)

        if i % 1000 == 0:
            alpha_vals.append(model.alpha.item())
            step_count.append(i)


    torch.save(model.state_dict(), "models/heat_inverse_pinn.pth")

    ACCENT, PAPER, HAIRLINE, INK_SOFT = "#B63679", "#FAFBFC", "#E2E6EA", "#3D4852"
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
    plt.plot(step_count, alpha_vals)
    plt.title('Inverse Problem')
    plt.xlabel('Run')
    plt.ylabel('Alpha')
    # plt.legend()
    plt.show()
    plt.savefig("figures/Alpha_inverse_problem.png", facecolor=PAPER, bbox_inches="tight", dpi=200)  

