# Creating my first MLP

import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SingleInputMLP(nn.Module):
    # SingleInputMLP is a child of the parent nn.Module and inherits all properties from it 
    def __init__(self):
        # this is the constructor, runs automatically once we create an instance (object) of the class. Its job is to set up the initial state of the object
        
        super(SingleInputMLP, self).__init__()
        # this tells the parent's constructor and telling it to run all the base/setup code before we start piling more things on
        # super - refers to parent class which in this case in nn.Module
        # .__init__() - calls parents constructor, teh setup code
        # SingleInputMLP, self tells super() which class we're in and which instance we are working with
        
        
        self.network = nn.Sequential(
            nn.Linear(1,4),
            nn.Tanh(),
            nn.Linear(4,1)
        )
        # nn.Sequential runs layers in order
        # input on top, passes through each layer one by one and then output comes out at the bottom
        # input --> nn.Linear() --> nn.ReU() --> nn.Linear() --> output
        # the numbers in nn.Linear are in_features and out_features and then it flips in the second nn.Linear()
        #  1 input produces 4 outputs and then the second nn.Linear takes in 4 inputs to produce 1 output
        # *** ReLU - is the Rectified Linear Unit. The activation function




    def forward(self,  x):
        # forward is the method Pytorch calls when we pass data through the model
        # we are defingin it here pytorch will call it will model(sample_input) runs
        return self.network(x)

model = SingleInputMLP()
# this creates an instance of our class and runs it and stores it in model
sample_input = torch.tensor([[1.0],[2.0],[3.0]], requires_grad=True)
# this creates a tensor with 3 inputs each with a single value. The shape in (3,1) - 3 samples, 1 feature\
# the model runs once for each, taking each one of them as an input
prediction = model(sample_input)
print(prediction)

# the current outputs are random they dont mean anything since the fucntion has random weights

# --- df/dx
first_grad = torch.autograd.grad(prediction, sample_input, grad_outputs=torch.ones_like(prediction), create_graph=True)
print(first_grad)
# syntax - torch.autograd.grad - is a function that computes derivatives
#  first argument is the output that we want to differentiate
#  second argument is what we are differentiating with respect to
#  the thrid argument is the weight for each output - here they are weighted equally

# remember to create_graph=True if you want to derive for a second time. If we dont then the program destroys the graph and we will not be able to differentiate again

h = 0.0001
# verify = (model(1+h) - model(1))/h -- model expects a tensor
verify = (model(torch.tensor([[1.0 + h]])) - model(torch.tensor([[1.0]])))/h
print(verify)

# --- d^2f/dx^2
print(torch.autograd.grad(first_grad, sample_input, grad_outputs=torch.ones_like(prediction)))

