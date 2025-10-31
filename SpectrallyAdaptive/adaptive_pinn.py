from SpectrallyAdaptive.adaptive_loss import AdaptiveLossWeighter, SpectralAnalyzer
import torch
import torch.nn as nn

class AdaptiveSpectralPINN(nn.Module):
    """Adaptive PINN with improved update rules and more frequent adaptation"""

    def __init__(self, layers=[2, 50, 50, 50, 1], nu=0.01, domain_bounds=[(-1, 1)]):
        super(AdaptiveSpectralPINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.nu = nu
        self.spectral_analyzer = SpectralAnalyzer(domain_bounds, n_bands=12)
        self.loss_weighter = AdaptiveLossWeighter(
            initial_weights={'pde': 1.0, 'bc': 1.0, 'ic': 1.0},
            adaptation_rate=0.05,
            max_weight=2.5,
            hf_threshold=0.01,
            time_scale=2000
        )

        self.history = {
            'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'ic_loss': [],
            'weights': [], 'spectral_errors': [], 'high_freq_ratios': []
        }

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

    def compute_pde_residual(self, x, u):
        u.requires_grad_(True)
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        u_x.requires_grad_(True)
        grad_u_x = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]

        residual = u_t + u * u_x - self.nu * u_xx
        return residual

    def compute_adaptive_loss(self, data_dict, epoch):
        x_colloc = data_dict['collocation']
        u_colloc = self.forward(x_colloc)
        residual = self.compute_pde_residual(x_colloc, u_colloc)
        pde_loss = torch.mean(residual**2)

        x_bc = data_dict['boundary_points']
        u_bc_pred = self.forward(x_bc)
        u_bc_true = data_dict['boundary_values'].unsqueeze(1)
        bc_loss = torch.mean((u_bc_pred - u_bc_true)**2)

        x_ic = data_dict['initial_points']
        u_ic_pred = self.forward(x_ic)
        u_ic_true = data_dict['initial_values'].unsqueeze(1)
        ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)

        if epoch % 20 == 0:
            with torch.no_grad():
                spectral_errors = self.spectral_analyzer.compute_spectral_errors(residual, x_colloc)
                current_losses = {'pde': pde_loss.item(), 'bc': bc_loss.item(), 'ic': ic_loss.item()}

                self.history['spectral_errors'].append(spectral_errors)
                total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
                high_freq_energy = sum([spectral_errors.get(f'band_{i}', 0.0) for i in range(8, 12)])
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                self.history['high_freq_ratios'].append(high_freq_ratio)

                pre_weights = self.loss_weighter.weights.copy()
                # print(f"[AdaptiveSpectralPINN] epoch={epoch} total_energy={total_energy:.6e} hf_ratio={high_freq_ratio:.6f} pre_weights={pre_weights}")

                self.loss_weighter.update_weights(spectral_errors, current_losses, epoch)

                post_weights = self.loss_weighter.weights.copy()
                # if post_weights != pre_weights:
                    # print(f"[AdaptiveSpectralPINN] epoch={epoch} POST-UPDATE weights={post_weights}")
        else:
            spectral_errors = self.history['spectral_errors'][-1] if self.history['spectral_errors'] else {}

        weights = self.loss_weighter.weights
        total_loss = (weights['pde'] * pde_loss + weights['bc'] * bc_loss + weights['ic'] * ic_loss)

        self.history['total_loss'].append(total_loss.item())
        self.history['pde_loss'].append(pde_loss.item())
        self.history['bc_loss'].append(bc_loss.item())
        self.history['ic_loss'].append(ic_loss.item())
        self.history['weights'].append(weights.copy())

        return total_loss
