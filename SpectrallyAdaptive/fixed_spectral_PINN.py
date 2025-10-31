from SpectrallyAdaptive.adaptive_loss import SpectralAnalyzer
import torch
import torch.nn as nn


class FixedAdaptivePINN(nn.Module):
    """Fixed version with incremental weight updates (debugging)"""
    def __init__(self, layers=[2, 50, 50, 50, 1], nu=0.01, domain_bounds=[(-1, 1)]):
        super(FixedAdaptivePINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.nu = nu
        self.spectral_analyzer = SpectralAnalyzer(domain_bounds, n_bands=12)

        self.weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
        self.base_weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0}
        self.adaptation_count = 0

        self.history = {
            'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'ic_loss': [],
            'weights': [], 'high_freq_ratios': [], 'adaptation_events': []
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

    def update_weights_incremental(self, high_freq_ratio, epoch):
        # print(f"  [FixedAdaptivePINN] Current weights: {self.weights}")
        # print(f"  [FixedAdaptivePINN] High-freq ratio: {high_freq_ratio:.6f}")

        old_weights = self.weights.copy()

        time_factor = 1.0 + min(epoch / 2000, 1.0)
        self.weights['pde'] = self.base_weights['pde'] * time_factor

        if high_freq_ratio > 0.01:
            hf_boost = high_freq_ratio * 2.0
            self.weights['pde'] += hf_boost
            # print(f"  🔥 High-freq boost: +{hf_boost:.3f}")

        self.weights['bc'] = max(self.base_weights['bc'] / time_factor, 0.2)
        self.weights['ic'] = max(self.base_weights['ic'] / time_factor, 0.2)

        total = sum(self.weights.values())
        for key in self.weights:
            # normalize to sum to 3
            self.weights[key] = self.weights[key] / (total + 1e-12) * 3.0

        weight_changed = any(abs(self.weights[key] - old_weights[key]) > 1e-6 for key in self.weights)
        if weight_changed:
            self.adaptation_count += 1
            # print(f"  ✅ WEIGHTS UPDATED: {old_weights} -> {self.weights}")
        else:
            pass
            # print(f"  ⚠️  Weights unchanged")

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
                total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
                high_freq_energy = sum([spectral_errors.get(f'band_{i}', 0.0) for i in range(len(spectral_errors)//2, len(spectral_errors))])
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                print(f"[FixedAdaptivePINN] epoch={epoch} total_energy={total_energy:.6e} hf_ratio={high_freq_ratio:.6f}")
                self.update_weights_incremental(high_freq_ratio, epoch)
                self.history['high_freq_ratios'].append(high_freq_ratio)
                self.history['adaptation_events'].append(self.adaptation_count)

        total_loss = (self.weights['pde'] * pde_loss + self.weights['bc'] * bc_loss + self.weights['ic'] * ic_loss)

        self.history['total_loss'].append(total_loss.item())
        self.history['pde_loss'].append(pde_loss.item())
        self.history['bc_loss'].append(bc_loss.item())
        self.history['ic_loss'].append(ic_loss.item())
        self.history['weights'].append(self.weights.copy())

        return total_loss
