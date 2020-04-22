import torch
import torch.optim
from torch import mm
from torch.optim.optimizer import Optimizer, required


class BFGS(Optimizer):
    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        self.H_k = torch.eye(params[0].shape[0])
        self.d_x = 0
        self.delta_k = torch.tensor(-1)
        self.first_iteration = True

        super(BFGS, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for x in group['params']:
                #print("x: "+str(x))
                if (self.delta_k**2).sum() ==0:
                    continue

                if x.grad is None:
                    continue

                # update H_k_new
                if self.first_iteration==True:
                    self.first_iteration = False


                else: #update H_k from BFGS updating rule
                    d_x_new = x.grad.data
                    d_x = self.d_x
                    delta_k = self.delta_k
                    delta_k = delta_k.reshape((delta_k.shape[0],1))
                    H_k = self.H_k
                    y_k = d_x_new - d_x
                    y_k = y_k.reshape((y_k.shape[0],1))


                    update_left = mm(H_k, mm(y_k, delta_k.t())) + mm(delta_k, mm(y_k.t(), H_k))
                    update_bottom = mm((mm(H_k, y_k)).t(), y_k)
                    update_left = update_left/update_bottom

                    belta_k = 1 + mm(y_k.t(), delta_k)/(mm(mm(H_k, y_k).t(), y_k))
                    update_right= mm(H_k, mm(y_k, mm(y_k.t(), H_k)))
                    update_right = update_right/update_bottom

                    delta_H_k = update_left - belta_k*update_right
                    self.H_k = H_k + delta_H_k

                    a = 0

                d_x = x.grad.data
                d_x = d_x.reshape((x.shape[0],1))
                d_x = d_x

                p_k = torch.mm(self.H_k,d_x)
                x.data.add_(-group['lr'], p_k)
                self.d_x = p_k
                self.delta_k = -group['lr']*p_k

        return loss