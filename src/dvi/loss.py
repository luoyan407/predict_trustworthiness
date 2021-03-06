import math
import torch
import torch.nn as nn



class GLLLoss(nn.Module):
    """Gaussian Log-Likelihood Loss
    """
    def __init__(self, style='heteroschedastic', method='bayes', homo_logvar_scale=0.4):
        super(GLLLoss, self).__init__()

        self.style = style
        self.method = method
        self.homo_logvar_scale = homo_logvar_scale
        self.gaussian_loglikelihood = (self.heteroschedastic_gaussian_loglikelihood
                                       if self.style == 'heteroschedastic'
                                       else self.homoschedastic_gaussian_loglikelihood)

    def forward(self, pred, target):
        log_likelihood = self.gaussian_loglikelihood(pred, target)
        return log_likelihood

    def extra_repr(self):
        return 'style={}, method={}{}'.format(self.style, self.method,
                                              ', scale={}'.format(self.homo_logvar_scale)
                                              if self.style != 'heteroschedastic' else '')

    def heteroschedastic_gaussian_loglikelihood(self, pred, target):
        log_variance = pred.mean[:, 1].reshape(-1)
        mean = pred.mean[:, 0].reshape(-1)

        if self.method.lower().strip() == 'bayes':
            sll = pred.var[:, 1, 1].reshape(-1)
            smm = pred.var[:, 0, 0].reshape(-1)
            sml = pred.var[:, 0, 1].reshape(-1)
        else:
            sll = torch.tensor(0.0).to(target.device)
            smm = torch.tensor(0.0).to(target.device)
            sml = torch.tensor(0.0).to(target.device)
        return self.gaussian_loglikelihood_core(target, mean, log_variance, smm, sml, sll)

    def homoschedastic_gaussian_loglikelihood(self, pred, target):
        log_variance = torch.tensor(self.homo_logvar_scale).to(target.device)
        mean = pred.mean[:, 0].reshape(-1)
        sll = torch.tensor(0.0).to(target.device)
        sml = torch.tensor(0.0).to(target.device)
        if self.method.lower().strip() == 'bayes':
            smm = pred.var[:, 0, 0].reshape(-1)
        else:
            smm = torch.tensor(0.0).to(target.device)
        return self.gaussian_loglikelihood_core(target, mean, log_variance, smm, sml, sll)

    def gaussian_loglikelihood_core(self, target, mean, log_variance, smm, sml, sll):
        return (-0.5 * (torch.tensor(math.log(2.0 * math.pi)) + log_variance
                        + torch.exp(-log_variance + 0.5 * sll) * (smm + (mean - sml - target) ** 2)))

def logsumexp(pred, keepdim=True, verbose=False):
    """
    compute <logsumexp(y)>
    """
    lse = torch.logsumexp(pred.mean, dim=-1, keepdim=keepdim)   # [b, 1]
    if verbose:
        print(lse.mean().item())
    p = torch.exp(pred.mean - lse)  # softmax                              # [b, y]
    diagonals = torch.diagonal(pred.var, offset=0, dim1=-2, dim2=-1) # [b, y]
    pTDiagVar = torch.sum(p * diagonals, dim=-1, keepdim=keepdim)        # [b, 1]
    # pTDiagVar = torch.sum(p * torch.matrix_diag_part(pred.var), dim=-1, keepdim=keepdim)        # [b, 1]
    pTVarp = torch.squeeze(torch.matmul(torch.unsqueeze(p, 1), torch.matmul(pred.var, torch.unsqueeze(p, 2))), dim=-1) # [b, 1]
    return lse + 0.5 * (pTDiagVar - pTVarp)

def logsoftmax(pred, verbose=False):
    """
    compute <logsoftmax(y)>
    """
    return pred.mean - logsumexp(pred, keepdim=True, verbose=verbose) # [b, y]

class CategoricalLoss(nn.Module):
    def __init__(self):
        super(CategoricalLoss, self).__init__()

    def forward(self, pred, target, verbose=True):
        # log_likelihood = self.gaussian_loglikelihood(pred, target)
        lsm = logsoftmax(pred, verbose=verbose)
        loss = torch.sum(target * lsm, dim=-1)

        return loss