import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # Everything above this line is kinda the boilerplate

        self.network = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(64,1)

            # the first linear takes in two inputs [position and time], the second linear outputs one value, that being temperature at that position and time
            # Using 64 neurons - as sepcified in the spec and we need 3 layers of Linear and Tanh
            # dropout zeros out the output of a neuron, Tanh is part of computing the output
            # Linear computes the value Tanh squashes it. Dropout decides whether to pass it on or zero it out.
        )

    def forward(self, x, t):

        combined = torch.cat((x, t), dim=1)
        # concatenating the two inputs into one tensor inside forward
        #  dim = 1 means we are stacking columns side by side
        # x tensor and t tensor both have shape (N,1), stacking them this way results in N samples each with 2 features
        # dim=0 would be stacking rows instead and we would get shape (2N, 1) which is wrong and not what nn.Sequential expects

        return self.network(combined)
    

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

# print(loss_pde(model))
# This returns a scalar but it means nothing just yet since the model in untrained, but we have the working mechanism

def loss_bc(model):
    x = torch.randint(low=0, high=2, size=(200,1)).float()
    # low is 0 and included however since high is excluded we need 1 to be included to we set high=2
    # size is how many points - shape of the tuples
    t = torch.rand(200, 1)

    u = model(x, t)
    L_bc = (u**2).mean()

    return L_bc

# print(loss_bc(model))

def loss_ic(model):
    t = torch.randint(low=0, high=1, size=(200, 1)).float()
    #  another way to write this t = 0 as this is the initial condition is:
    # t = torch.zeros(200,1) ---> zero across teh entire array just like above
    x = torch.rand(200, 1)
    u_ic = torch.sin(torch.pi*x)

    u = model(x,t)
    loss_ic = ((u_ic - u)**2).mean()
    return loss_ic

# print(loss_ic(model))

def mc_dropout_predict(model, x, t, n_passes):
    model.train()
    # switches to training mode
    results = []
    # its an empty list where the results stay
    for i in range(n_passes):
        res = model(x,t).detach()
        # detach returns a new tensor that is disconnected from the computational graph
        # when we call detach it stops tracking future operations on that one tensor
        results.append(res)
    
    final_res = torch.stack(results)
    mean = final_res.mean(dim=0) 
    std = final_res.std(dim=0)
    return mean, std

if __name__ == "__main__":
    model = PINN()
    sample_t = torch.tensor([[2.0],[0.4],[0.5]], requires_grad=True)
    sample_x = torch.tensor([[1.2],[3.2],[5.0]], requires_grad=True)
    prediction = model(sample_x, sample_t)
    print(prediction)

    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # This is instantiating an optimizer
    # adam is an optimizer whose job is to update the weights in order to reduce loss
    # nn.Linear computes y = W*x + b (W is a matrix of weights and b is baises)
    # Adam's job is to update W and b in every nn.Linear layer after each step
    # nn.Linear(2, 64) takes 2 inputs (x and t) ad outputs 64 numbers
    # the 64 output numbers has W a 64x2 matrix of weights. Each of teh outputs is a weighted sum of the 2 inputs, b is a vector of 64 biases, one added to each output
    # Adam's job is to look at the loss and nudge every single number in W and b slightly so the loss gets smaller

    # What are model.parameters()??
    # They are just all the W matrices and b vectors inside the 4 layers of nn.Linear
    # We present adam with the numbers it is allowed to update and it decides the direction to nudge the weights in

    # lr is the learning rate, its a hyperparameter that controls the step size the optimizer takes when updating model weights

    for i in range(10000):
        optimizer.zero_grad()
        # This basically zeros each step's gradients that would normally pile onto the previous step's
        # pytorch accumulates gradients by default this mitigates the problem of not being able to compare gradients across steps
        
        loss1 = loss_pde(model)
        loss2 = loss_bc(model)
        loss3 = loss_ic(model) 
        # we are saving the losses in variables so we lock it in which prevents us from using one and printing something else down below

        total_loss = loss1 + loss2 + loss3
        total_loss.backward()
        # Forward passes are input -> layers -> prediction -> loss
        # backward() goes in the other direction, walks back every opeation and computes how much did each weight contribute to the loss
        # that number is the gradient adam then uses in .step() to nudge each weight in the direction that reduces the loss

        optimizer.step()
        if i % 1000 == 0:
            print(loss1)
            print(loss2)
            print(loss3)
            print("-----------")
            print(total_loss)
            print("***********")

        # the optimizer is teaching the model to optimize and get the loss as close to 0 as possible
        # This leads to satisfying the equation which is what we need

    torch.save(model.state_dict(), "models/heat_pinn.pth")

    x_new = torch.linspace(0, 1, 100).reshape(100, 1)
    t_new = torch.full((100, 1), 0.5)
    # we need both to be the same size that being 100 rows and 1 column
    mean, std = mc_dropout_predict(model, x_new, t_new, 200)

    plt.plot(x_new.numpy().flatten(), mean.numpy().flatten())
    plt.fill_between(x_new.numpy().flatten(), (mean - 2*std).numpy().flatten(), (mean + 2*std).numpy().flatten(), color="blue")
    plt.xlabel("x")
    plt.ylabel("u(x, t=0.5)")
    plt.title("MC Dropout Uncertainty @ t = 0.5")

    plt.savefig("figures/MC_dropout_t_0.5.png")
    plt.show()

