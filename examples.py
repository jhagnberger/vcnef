import torch

from vcnef.vcnef_1d import VCNeFModel as VCNeF1DModel
from vcnef.vcnef_2d import VCNeFModel as VCNeF2DModel
from vcnef.vcnef_3d import VCNeFModel as VCNeF3DModel
from utils.utils import count_model_params


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example for 1D PDEs
model = VCNeF1DModel(num_channels=3,
                     condition_on_pde_param=True,
                     pde_param_dim=2,
                     d_model=96,
                     n_heads=8,
                     n_transformer_blocks=3,
                     n_modulation_blocks=3)
model = model.to(device)
print("1D model parameter count:", count_model_params(model))

# Random data with shape b, s, c
x = torch.rand(4, 256, 3, device=device)
grid = torch.rand(4, 256, 1, device=device)
pde_param = torch.rand(4, 2, device=device)
t = torch.arange(1, 21, device=device).repeat(4, 1) / 20

y_hat = model(x, grid, pde_param, t)
print("1D output:", y_hat.shape)


# Example for 2D PDEs
model = VCNeF2DModel(num_channels=4,
                     condition_on_pde_param=False,
                     pde_param_dim=2,
                     d_model=256,
                     n_heads=8,
                     n_transformer_blocks=1,
                     n_modulation_blocks=6)
model = model.to(device)
print("2D model parameter count:", count_model_params(model))

# Random data with shape b, s_x, s_y, c
x = torch.rand(4, 64, 64, 4, device=device)
grid = torch.rand(4, 64, 64, 2, device=device)
pde_param = torch.rand(4, 2, device=device)
t = torch.arange(1, 21, device=device).repeat(4, 1) / 20

y_hat = model(x, grid, pde_param, t)
print("2D output:", y_hat.shape)


# Example for 3D PDEs
model = VCNeF3DModel(num_channels=5,
                     condition_on_pde_param=False,
                     pde_param_dim=2,
                     d_model=256,
                     n_heads=8,
                     n_transformer_blocks=1,
                     n_modulation_blocks=6)
model = model.to(device)
print("3D model parameter count:", count_model_params(model))

# Random data with shape b, s_x, s_y, s_z, c
x = torch.rand(4, 32, 32, 32, 5, device=device)
grid = torch.rand(4, 32, 32, 32, 3, device=device)
pde_param = torch.rand(4, 2, device=device)
t = torch.arange(1, 21, device=device).repeat(4, 1) / 20

y_hat = model(x, grid, pde_param, t)
print("3D output:", y_hat.shape)
