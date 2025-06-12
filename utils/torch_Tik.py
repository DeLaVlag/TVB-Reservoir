import numpy as np
import torch
from sympy import pprint_use_unicode
import matplotlib.pyplot as plt
from torch.linalg import inv



def torch_matrices(R, Y):
# Assuming you have existing NumPy matrices 'R' and 'Y'
    R = np.random.rand(200, 68, 16)
    Y = np.random.rand(16, 3, 200)

    # Transfer the NumPy matrices to PyTorch tensors and move them to the GPU
    Rtorch = torch.from_numpy(R).cuda()
    Ytorch = torch.from_numpy(Y).cuda()

    # Matrix multiplication using einsum
    # Z = torch.einsum('ijk,lmi->lim', Rtorch, Ytorch)

    Z = torch.einsum('ijk,lmi->kjm', Rtorch, Ytorch)

    print("Shape of the result tensor Z:", Z)

# torch_matrices()

def Tik_torch(R, Y, beta_ridge):

    # torch those nparrs
    R = torch.from_numpy(R).float().cuda()
    Y = torch.from_numpy(Y).float().cuda()

    # rule is to have the batch sizes so the nworkitems at the first index. then first need to have last and 2nd
    # need to have the middle indices equal. ex [10, (4), 4] * [10, 4, (2)] = [10, (4), (2)]

    # tensor2 = torch.randn(10, 5, 4).cuda()
    RT = R.permute(0, 2, 1).cuda() # (16, 200, 68)
    # print("RT.shape", RT.shape)
    RTR = torch.matmul(R, RT)
    # print(f"Min value of RTR: {torch.min(RTR).item()}")
    # print(f"Max value of RTR: {torch.max(RTR).item()}")
    # print(RTR.shape) # (10, 4, 4) or (16, 68, 68)
    # eye = torch.eye(RTR.size(1), device=RTR.device).unsqueeze(2)
    eye = torch.eye(RTR.size(1), device=RTR.device).unsqueeze(0).expand(RTR.size(0), RTR.size(1), RTR.size(2))
    R_T_R_inv = inv(RTR + beta_ridge * eye)
    # print("R_T_R", R_T_R_inv.shape) # (10, 4, 4)
    R_T_R_inv = R_T_R_inv.permute((1, 2, 0))
    # print("R_T_Rinv", R_T_R_inv.shape) # (4, 4, 10) (68, 68, 16)
    RT = RT.permute(0, 2, 1)
    # print("Y", Y.shape)
    # print("RT", RT.shape)
    R_T_Y = torch.matmul(RT, Y) #(10,5,4) * (10,2,5)    16 62 200 vs 16 25 3
    # print("R_T_Y", R_T_Y.shape)
    R_T_R_inv = R_T_R_inv.permute(2, 0, 1)
    # print('R_T_R_invp', R_T_R_inv.shape)
    R_T_Y = R_T_Y.permute(0, 1, 2)
    # print('R_T_Yp', R_T_Y.shape)
    Z = torch.matmul(R_T_R_inv, R_T_Y) # ([4, 4, 10]) ([10, 4, 2])     Shape: (16, 68, 3)

    Wout = Z.cpu().numpy()
    # print("isnan((Wout)", np.isnan((Wout).any()))

    return Wout

def predi_Torch_norm(R, Y, W_out):

    '''
    shapes we want:
    R [16, 62, 200]
    Y [16, 200, 3]
    Wout [16, 62, 3]

    This computes only the mse and does not correct it.
    Many high outcomes
    '''

    R = torch.from_numpy(R).float().cuda()
    Y = torch.from_numpy(Y).float().cuda()
    W_out = torch.from_numpy(W_out).float().cuda()

    # print("R.shape", R.shape) # [16, 68, 200]
    # print("Y.shape", Y.shape) # [16, 200, 3]
    # print("W_out.shape", W_out.shape) # [16, 68, 3]
    # print("W_outp.shape", R.permute(0, 2, 1).shape) # 16, 3, 68

    # Perform batched matrix multiplication: (n_work_items, 200, 68) @ (n_work_items, 68, 3) -> (n_work_items, 200, 3)
    Y_pred = torch.bmm(R.permute(0, 2, 1), W_out)  # Resulting shape: (n_work_items, 200, 3)
    # print("Y_pred.shape", Y_pred.shape) # 16, 200, 3


    # Calculate mean squared error for each work item
    pureMSE = torch.mean((Y.transpose(1, 2) - Y_pred.permute(0, 2, 1)) ** 2, dim=(1, 2))  # Shape: (n_work_items,)

    # `Y_pred` now contains all predictions, and `pureMSE` contains the MSE for each work item
    return pureMSE.cpu().numpy(), Y_pred.cpu().numpy()


def predi_Torch_msexpdecaycorr(R, Y, W_out, T=100):
    '''
    shapes we want:
    R [16, 62, 200]
    Y [16, 200, 3]
    Wout [16, 62, 3]

    This computes only the mse and does not correct it.
    Many high outcomes
    '''

    n_work_items = R.shape[0]
    # T=100

    Rto = torch.from_numpy(R).float().cuda()
    Yto = torch.from_numpy(Y).float().cuda()
    W_out = torch.from_numpy(W_out).float().cuda()

    print("R.shape", Rto.shape)  # [16, 62, 100]
    print("Y.shape", Yto.shape)  # [16, 100, 3]
    print("W_out.shape", W_out.shape)  # [16, 68, 3]
    print("Rto.p.shape", Rto.permute(0, 2, 1).shape)  # 16, 100, 62

    # Perform batched matrix multiplication: (n_work_items, 100, 62) @ (n_work_items, 62, 3) -> (n_work_items, 100, 3)
    Y_pred = torch.bmm(Rto.permute(0, 2, 1), W_out)  # ([16, 100, 62]) @ ([16, 62, 3]) Resulting shape: (n_work_items, 100, 3)
    print("Y_pred.shape", Y_pred.shape)  # 16, 100, 3

    n_samples = Rto.shape[2]
    # n_samples = 100

    t = torch.arange(n_samples, device="cuda:0").float()
    # print("n_samples", n_samples, "t.shape", t.shape), only till 100 to evade the transient in pre
    exp_decay = torch.exp(-t / T)
    exp_decay /= exp_decay.sum()
    # print("exp_decay", exp_decay)

    # error = (Yto.transpose(1, 2)[:,:,20:120] - Y_pred.permute(0, 2, 1)[:,:,20:120]) ** 2
    error = (Yto.transpose(1, 2) - Y_pred.permute(0, 2, 1)) ** 2
    weighted_error = error * exp_decay # error[16, 3, 100] * [1, 200, 1]
    # print("weighted_error.shape", weighted_error.shape)

    # Calculate mean squared error for each work item
    pureMSE = torch.mean(weighted_error, dim=(1, 2))  # Shape: (n_work_items,)

    # Compute weighted variance of Y (scaled similarly to MSE)
    mean_Y = torch.mean(Yto, dim=(1, 2), keepdim=True)  # [16, 1, 1]
    var_Y = torch.mean((Yto - mean_Y) ** 2, dim=(1, 2))  # Shape: (n_work_items,)

    # Normalize MSE
    normalized_MSE = (pureMSE / var_Y)

    normalized_MSE = normalized_MSE.cpu().numpy()
    pureMSE = pureMSE.cpu().numpy()
    Y_pred = Y_pred.cpu().numpy()

    corrected_mse_percentage = (normalized_MSE / np.sum(normalized_MSE))

    # `Y_pred` now contains all predictions, and `pureMSE` contains the MSE for each work item
    return pureMSE, Y_pred, corrected_mse_percentage


def predi_Torch_mse_exdecay_plus(R, Y, W_out, T=100, alpha=0.1, beta=0.01, epsilon=1e-3):
    '''
    Computes the weighted MSE with additional penalties and corrections.
    Shapes:
    R: [n_work_items, n_regions, n_timesteps] e.g., [16, 62, 200]
    Y: [n_work_items, n_timesteps, n_outputs] e.g., [16, 200, 3]
    W_out: [n_work_items, n_regions, n_outputs] e.g., [16, 62, 3]
    '''
    n_work_items = R.shape[0]

    Rto = torch.from_numpy(R).float().cuda()
    Yto = torch.from_numpy(Y).float().cuda()
    W_out = torch.from_numpy(W_out).float().cuda()

    print("R.shape", Rto.shape)  # [16, 62, 200]
    print("Y.shape", Yto.shape)  # [16, 200, 3]
    print("W_out.shape", W_out.shape)  # [16, 62, 3]

    # Compute predicted output
    Y_pred = torch.bmm(Rto.permute(0, 2, 1), W_out)  # Shape: [16, 200, 3]

    # Exponential decay weights
    n_samples = Rto.shape[2]
    t = torch.arange(n_samples, device="cuda:0").float()
    exp_decay = torch.exp(-t / T)
    exp_decay /= exp_decay.sum()

    # Compute weighted MSE
    error = (Yto.transpose(1, 2) - Y_pred.permute(0, 2, 1)) ** 2
    weighted_error = error * exp_decay  # Shape: [16, 3, 200]

    # Add gradient penalty
    gradient_error = (torch.diff(Yto, dim=1) - torch.diff(Y_pred, dim=1)) ** 2  # [16, 199, 3]
    weighted_gradient_error = gradient_error * exp_decay[:-1].view(1, -1, 1) # Match dimensions for broadcasting
    total_gradient_error = alpha * torch.mean(weighted_gradient_error, dim=(1, 2))  # Shape: [16]

    # Add variance penalty
    predicted_variance = torch.var(Y_pred, dim=1)  # Variance across time: Shape [16, 3]
    variance_penalty = beta / (predicted_variance + 1e-6)  # Avoid division by zero
    total_variance_penalty = torch.mean(variance_penalty, dim=1)  # Shape: [16]

    # Calculate pure MSE
    pureMSE = torch.mean(weighted_error, dim=(1, 2))  # Shape: [16]

    # Compute weighted variance of Y
    mean_Y = torch.mean(Yto, dim=(1, 2), keepdim=True)  # [16, 1, 1]
    var_Y = torch.mean((Yto - mean_Y) ** 2, dim=(1, 2)) + epsilon  # Add epsilon for stability

    # Total error with penalties
    total_error = pureMSE + total_gradient_error + total_variance_penalty

    # Normalize MSE
    normalized_MSE = total_error / var_Y

    # Convert results to numpy
    normalized_MSE = normalized_MSE.cpu().numpy()
    pureMSE = pureMSE.cpu().numpy()
    Y_pred = Y_pred.cpu().numpy()

    # Calculate percentage correction for normalized MSE
    corrected_mse_percentage = (normalized_MSE / np.sum(normalized_MSE))

    # Return pure MSE, predictions, and normalized MSE percentage
    return pureMSE, Y_pred, corrected_mse_percentage


def predi_Torch(R, W_out):
    '''
    shapes we want:
    R [16, 62, 10]
    Wout [16, 62, 3]

    Just prediction for the feedback loop
    '''

    n_work_items = R.shape[0]
    # T=100

    R = torch.from_numpy(R).float().cuda()
    W_out = torch.from_numpy(W_out).float().cuda()

    # print("R.shape", R.shape)  # [16, 62, 100]
    # print("W_out.shape", W_out.shape)  # [16, 68, 3]
    # print("Rto.p.shape", Rto.permute(0, 2, 1).shape)  # 16, 100, 62

    # Perform batched matrix multiplication: (n_work_items, 100, 62) @ (n_work_items, 62, 3) -> (n_work_items, 100, 3)
    Y_pred = torch.bmm(R, W_out)  # ([16, 100, 62]) @ ([16, 62, 3]) Resulting shape: (n_work_items, 100, 3)
    # print("Y_pred.shape", Y_pred.shape)  # 16, 100, 3

    return Y_pred.cpu().numpy()


if __name__ == "__main__":


    # R = torch.randn(16, 68, 200).cuda() # (16, 68, 200)
    # Y = torch.rand(16, 200, 3).cuda()  # (nsims, output_dim, nts) -> (nts, output_dim, nsims)

    # R = np.random.randn(16, 68, 200) # (16, 68, 200)
    # Y = np.random.randn(16, 200, 3)  # (nsims, output_dim, nts) -> (nts, output_dim, nsims)

    def generate_sinusoidal_data(nsims, regions, timesteps, outputs):
        freq = 2 * np.pi * np.linspace(0, 1, timesteps)
        freq2 = 65 * np.pi * np.linspace(0, 1, timesteps)

        # Generate R with sinusoidal signals + noise, shape (nsims, regions, timesteps)
        R = np.zeros((nsims, regions, timesteps))
        for sim in range(nsims):
            for region in range(regions):
                R[sim, region, :] = np.sin(freq + region * 0.1) #+ np.random.rand(timesteps) * 0.1  # Noise for each region

        # Generate Y with sinusoidal signals + random integer offset, shape (nsims, timesteps, outputs)
        Y = np.zeros((nsims, timesteps, outputs))
        for sim in range(nsims):
            for output in range(outputs):
                Y[sim, :, output] = np.cos(freq2 + output * 0.1) # + np.random.randint(-10, 10, size=timesteps)  # Offset for each output

        return R, Y

    # Generate sinusoidal signals for best_mse_tloop0 and best_mse_tloop1
    # timesteps = 64
    # freq = 2 * np.pi * np.linspace(0, 1, timesteps)
    # R = np.sin(freq) + np.random.rand(1, timesteps) * 0.1  # Sinusoidal with noise
    # Y = np.cos(freq) + np.random.randint(-10, 10, size=(1, timesteps))  # Sinusoidal with random integer offset

    R, Y = generate_sinusoidal_data(16, 62, 200, 3)

    Z= Tik_torch(R, Y, beta_ridge=0.01)
    # MSEs = predi_Torch(R, Y, Z)
    # MSEs, Ypred, CR = predi_Torch_msexpdecaycorr(R, Y, Z, 1)
    MSEs, Ypred, CR = predi_Torch_mse_exdecay_plus(R, Y, Z)

    print("MSEs", MSEs[np.argsort(MSEs)])