import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb

# Import PDEBench dataloader
from utils.dataset import PDEBenchDataset

# Import VCNeF models
from vcnef.vcnef_1d import VCNeFModel as VCNeF1DModel
from vcnef.vcnef_2d import VCNeFModel as VCNeF2DModel
from vcnef.vcnef_3d import VCNeFModel as VCNeF3DModel

# Import function for counting model trainable parameters
from utils.utils import count_model_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training():
    """
    This training loop is an adapted version of the PDEBench training loop.
    """

    base_path = "pdebench/data/"
    file_names = ["1D_Burgers_Sols_Nu0.001.hdf5"]
    num_channels = 1
    pde_param_dim = 1
    condition_on_pde_param = False
    t_train = 41
    initial_step = 1
    reduced_resolution = 4
    reduced_resolution_t = 5
    reduced_batch = 1

    num_workers = 8
    model_update = 1
    model_path = "VCNeF.pt"

    batch_size = 32
    epochs = 500
    learning_rate = 3.e-4
    random_seed = 3407

    scheduler_warmup_fraction = 0.2

    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Initialize W&B
    wandb.init()

    # Initialize the dataset and dataloader
    train_data = PDEBenchDataset(file_names,
                                 reduced_resolution=reduced_resolution,
                                 reduced_resolution_t=reduced_resolution_t,
                                 reduced_batch=reduced_batch,
                                 initial_step=initial_step,
                                 saved_folder=base_path)
    val_data = PDEBenchDataset(file_names,
                               reduced_resolution=reduced_resolution,
                               reduced_resolution_t=reduced_resolution_t,
                               reduced_batch=reduced_batch,
                               initial_step=initial_step,
                               if_test=True,
                               saved_folder=base_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print(f"shape of train_data, 0: {train_data[-1][0].shape}")
    print(f"shape of train_data, 1: {train_data[-1][1].shape}")
    print(f"shape of train_data, 2: {train_data[-1][2].shape}")
    print(f"shape of train_data, 3: {train_data[-1][3].shape}")
    print(f"shape of val_data, 0: {val_data[-1][0].shape}")
    print(f"shape of val_data, 1: {val_data[-1][1].shape}")
    print(f"shape of val_data, 2: {val_data[-1][2].shape}")
    print(f"shape of val_data, 3: {val_data[-1][3].shape}")
    print(f"length of train_loader: {len(train_loader)}")
    print(f"length of val_loader: {len(val_loader)}")

    _, _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print("Spatial Dimension", dimensions - 3)

    # Set up model
    if dimensions == 4:
        print("VCNeF 1D")
        model = VCNeF1DModel(d_model=96,
                            n_heads=8,
                            num_channels=num_channels,
                            condition_on_pde_param=condition_on_pde_param,
                            pde_param_dim=pde_param_dim,
                            n_transformer_blocks=3,
                            n_modulation_blocks=3)
    elif dimensions == 5:
        print("VCNeF 2D")
        model = VCNeF2DModel(d_model=256,
                             n_heads=8,
                             num_channels=num_channels,
                             condition_on_pde_param=condition_on_pde_param,
                             pde_param_dim=pde_param_dim,
                             n_transformer_blocks=1,
                             n_modulation_blocks=6)
    elif dimensions == 6:
        print("VCNeF 3D")
        model = VCNeF3DModel(d_model=256,
                             n_heads=8,
                             num_channels=num_channels,
                             condition_on_pde_param=condition_on_pde_param,
                             pde_param_dim=pde_param_dim,
                             n_transformer_blocks=1,
                             n_modulation_blocks=6)
    model.to(device)
    total_params = count_model_params(model)
    print(f"Total Trainable Parameters = {total_params}")

    # Set maximum time step of the data to train
    if t_train > _data.shape[-2]:
        t_train = _data.shape[-2]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=learning_rate,
                                                    pct_start=scheduler_warmup_fraction,
                                                    div_factor=1e3,
                                                    final_div_factor=1e4,
                                                    total_steps=epochs * len(train_loader))

    loss_fn = nn.MSELoss(reduction="mean")
    loss_fn_no_reduction = nn.MSELoss(reduction="none")
    loss_val_min = np.infty

    for ep in range(epochs):
        model.train()
        train_l2_step = 0
        train_l2_full = 0
        train_l2_full_mean = 0

        for xx, yy, grid, pde_param in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pde_param = pde_param.to(device)

            yy = yy[..., 0:t_train, :]

            # Prepare queried times t in [0..1]
            t = torch.arange(initial_step, t_train, device=xx.device) * 1 / (t_train-1)
            t = t.repeat((xx.size(0), 1))

            # Forward pass
            pred = model(xx[..., 0, :], grid, pde_param, t)
            pred = torch.cat((xx, pred), dim=-2)

            # Loss calculation
            _batch = yy.size(0)
            loss = torch.sum(torch.mean(loss_fn_no_reduction(pred.unsqueeze(-1), yy.unsqueeze(-1)), dim=(0, 1)))
            l2_full = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
            train_l2_step += loss.item()
            train_l2_full += l2_full
            train_l2_full_mean += l2_full * _batch

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            scheduler.step()

        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            val_l2_full_mean = 0
            model.eval()

            with torch.no_grad():
                for xx, yy, grid, pde_param in val_loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
                    grid = grid.to(device)
                    pde_param = pde_param.to(device)

                    # Prepare queried times t in [0..1]
                    t = torch.arange(initial_step, yy.shape[-2], device=xx.device) * 1 / (t_train-1)
                    t = t.repeat((xx.size(0), 1))

                    # Forward pass
                    pred = model(xx[..., 0, :], grid, pde_param, t)
                    pred = torch.cat((xx, pred), dim=-2)

                    # Loss calculation
                    _batch = yy.size(0)
                    loss = torch.sum(torch.mean(loss_fn_no_reduction(pred.unsqueeze(-1), yy.unsqueeze(-1)), dim=(0, 1)))
                    l2_full = loss_fn(pred.reshape(_batch, -1), yy.reshape(_batch, -1)).item()
                    val_l2_step += loss.item()
                    val_l2_full += l2_full
                    val_l2_full_mean += l2_full * _batch

                # Calculate mean of l2 full loss
                train_l2_full_mean = train_l2_full_mean / len(train_loader.dataset)
                val_l2_full_mean = val_l2_full_mean / len(val_loader.dataset)

                # Save checkpoint
                if val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    torch.save({
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss_val_min
                    }, model_path)
            model.train()

        # Log metrics in W&B
        wandb.log({
            "train/loss": train_l2_full,
            "train/mean_loss": train_l2_full_mean,
            "val/loss": val_l2_full,
            "val/mean_loss": val_l2_full_mean,
            "lr": scheduler.get_last_lr()[0]
        })


if __name__ == "__main__":
    run_training()
    print("Done.")
