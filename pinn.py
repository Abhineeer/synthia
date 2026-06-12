import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # Everything above this line is kinda the boilerplate

        self.network = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64,1)

            # the first linear takes in two inputs [position and time], the second linear outputs one value, that being temperature at that position and time
            # Using 64 neurons - as sepcified in the spec and we need 3 layers of Linear and Tanh
        )

    def forward(self, x, t):

        combined = torch.cat((x, t), dim=1)
        # concatenating the two inputs into one tensor inside forward
        #  dim = 1 means we are stacking columns side by side
        # x tensor and t tensor both have shape (N,1), stacking them this way results in N samples each with 2 features
        # dim=0 would be stacking rows instead and we would get shape (2N, 1) which is wrong and not what nn.Sequential expects

        return self.network(combined)
    

model = PINN()
sample_t = torch.tensor([[2.0],[0.4],[0.5]], requires_grad=True)
sample_x = torch.tensor([[1.2],[3.2],[5.0]], requires_grad=True)
prediction = model(sample_x, sample_t)
print(prediction)

# Our PINN works
# the weights are random so teh numbers that come out the other end are useless until we implement our loss function


def loss_pde(model):
    x = torch.rand(2000, 1, requires_grad=True)
    t = torch.rand(2000, 1, requires_grad=True)
    # torch.rand uses the Uniform distribution and the values only range from 0 to 1
    u = model(x,t)
    ut_firstpde = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)
    ux_firstpde = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    ux_secondpde = torch.autograd.grad(ux_firstpde[0], x, grad_outputs=torch.ones_like(ux_firstpde[0]), create_graph=True)
    # ux_firstpde[0], we use insted of simply writing ux_fristpde because ux_firstpde is a tuple and because torch.autograd.grad returns a tuple
    # even if we are differentiating with respect to one tensor ==> so the actual gradiet tensor we get from differentiating becomes the one and only element of the tuple
    # the tuple reads like this -> (tensor([...]),)
    # the tensor is at index 0 of teh tuple and we call it by using ux_firstpde[0]

    alpha = 0.1

    L_pde = ((ut_firstpde[0] - alpha*ux_secondpde[0])**2).mean()
    # mean squared error process

    return L_pde

print(loss_pde(model))