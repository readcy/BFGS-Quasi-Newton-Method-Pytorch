# BFGS-Quasi-Newton-Method-Pytorch

This is an Pytorch implementation of BFGS Quasi Newton Method optimization algorithm. You can just import BFGS in your file and use it as other optimizers you use in Pytorch.

'''
from BFGS import BFGS

optimizer = torch.optim.BFGS(model.parameters(), lr=0.1)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
'''