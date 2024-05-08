import torch

def count_model_params(model):
    """
    Calculates number of parameters of given model. Counts real
    and imaginary part of complex-valued weights and biases.
    """
    params = []
    for p in model.parameters():
        if p.requires_grad:
            if torch.is_complex(p):
                params.append(2 * p.numel())
            else:
                params.append(p.numel())
    return sum(params)
