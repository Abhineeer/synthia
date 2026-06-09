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

