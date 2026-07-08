import torch
import torch.nn as nn



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

x_obs = torch.rand(150, 1)
t_obs = torch.rand(150, 1)
# torch.rand - 150 random points of x and t, [0, 1)
    
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
