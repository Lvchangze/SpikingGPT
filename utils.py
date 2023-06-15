import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from spikingjelly.clock_driven import functional


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):

    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        functional.reset_net(model)

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        
        
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
           
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # return probs


        
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
    
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
        # return x

    return x



