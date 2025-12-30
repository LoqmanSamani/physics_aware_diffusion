def energy_fp_loss_graph(residual, alpha = 1.0):
    return alpha * (residual ** 2).mean()
