import numpy as np
from scipy import fft
class SpectralAnalyzer:
    """Enhanced spectral analysis for PINNs"""

    def __init__(self, domain_bounds, n_bands=12):
        self.domain_bounds = domain_bounds
        self.n_bands = n_bands
        self.frequency_bands = self._initialize_frequency_bands()

    def _initialize_frequency_bands(self):
        domain_size = self.domain_bounds[0][1] - self.domain_bounds[0][0]
        max_freq = 1.0 / (2.0 * (domain_size / 50))
        low_freqs = np.logspace(-1, np.log10(max_freq), self.n_bands)
        high_freqs = np.logspace(-1, np.log10(max_freq), self.n_bands + 1)[1:]
        bands = [(low, high) for low, high in zip(low_freqs, high_freqs)]
        return bands

    def compute_spectral_errors(self, residual, coordinates):
        x = coordinates[:, 0].detach().cpu().numpy()
        res_vals = residual.detach().cpu().numpy().ravel()

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        res_sorted = res_vals[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        res_unique = res_sorted[unique_idx]

        x_unique = np.asarray(x_unique).ravel()
        res_unique = np.asarray(res_unique).ravel()

        if x_unique.size < 2:
            return {f'band_{i}': 0.0 for i in range(self.n_bands)}

        n_points = min(256, len(x_unique))
        x_uniform = np.linspace(float(x_unique.min()), float(x_unique.max()), n_points)
        res_uniform = np.interp(x_uniform, x_unique, res_unique)

        fft_vals = fft.fft(res_uniform)
        freqs = fft.fftfreq(len(res_uniform), x_uniform[1] - x_uniform[0])

        spectral_energy = {}
        total_energy = np.sum(np.abs(fft_vals)**2) + 1e-12
        for i, (low, high) in enumerate(self.frequency_bands):
            mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
            band_energy = np.sum(np.abs(fft_vals[mask])**2)
            spectral_energy[f'band_{i}'] = float(band_energy / total_energy)

        return spectral_energy

    def compute_spectral_energies_absolute(self, residual, coordinates):
        """Return absolute band energies (not normalized) and the total energy.

        Returns:
          (band_dict, total_energy)
        where band_dict maps 'band_i' -> absolute energy (float).
        """
        x = coordinates[:, 0].detach().cpu().numpy()
        res_vals = residual.detach().cpu().numpy().ravel()

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        res_sorted = res_vals[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        res_unique = res_sorted[unique_idx]

        x_unique = np.asarray(x_unique).ravel()
        res_unique = np.asarray(res_unique).ravel()

        if x_unique.size < 2:
            return ({f'band_{i}': 0.0 for i in range(self.n_bands)}, 0.0)

        n_points = min(256, len(x_unique))
        x_uniform = np.linspace(float(x_unique.min()), float(x_unique.max()), n_points)
        res_uniform = np.interp(x_uniform, x_unique, res_unique)

        fft_vals = fft.fft(res_uniform)
        freqs = fft.fftfreq(len(res_uniform), x_uniform[1] - x_uniform[0])

        spectral_energy = {}
        total_energy = float(np.sum(np.abs(fft_vals)**2) + 1e-12)
        for i, (low, high) in enumerate(self.frequency_bands):
            mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
            band_energy = float(np.sum(np.abs(fft_vals[mask])**2))
            spectral_energy[f'band_{i}'] = band_energy

        return spectral_energy, total_energy


class AdaptiveLossWeighter:
    """Incremental adaptive loss weighter with configurable behavior and logging."""

    def __init__(self, initial_weights, adaptation_rate=0.05, min_weight=0.1, max_weight=3.0, hf_threshold=0.01, time_scale=2000):
        self.weights = {k: float(v) for k, v in initial_weights.items()}
        self.base_weights = {k: float(v) for k, v in initial_weights.items()}
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.hf_threshold = hf_threshold
        self.time_scale = time_scale
        self.error_history = []

    def update_weights(self, spectral_errors, current_losses, epoch=0):
        total_energy = sum(spectral_errors.values()) if spectral_errors else 0.0
        n_bands = len(spectral_errors) if spectral_errors else 0
        if n_bands == 0:
            return

        high_start = n_bands // 2
        high_energy = sum(spectral_errors.get(f'band_{i}', 0.0) for i in range(high_start, n_bands))
        high_ratio = high_energy / (total_energy + 1e-12)

        old_weights = self.weights.copy()

        time_factor = 1.0 + min(epoch / max(1, self.time_scale), 1.0)

        # incorporate the current PDE loss into the PDE weight: we want to
        # reduce the PDE loss toward zero, so include the PDE loss value
        # (from current_losses) into the weight baseline before other tweaks.
        pde_loss_val = float(current_losses.get('pde', 0.0))
        # pde_val is the base weight magnitude: pde_loss + time factor
        # pde_val = pde_loss_val + time_factor
        pde_val = pde_loss_val

        # apply an HF-driven small boost if the high-frequency ratio is above threshold
        if high_ratio > self.hf_threshold:
            hf_boost = high_ratio * 2.0 * self.adaptation_rate * (1.0 + time_factor - 1.0)
            pde_val = min(pde_val + hf_boost, self.max_weight)

        bc_val = max(self.base_weights.get('bc', 1.0) / time_factor, self.min_weight)
        ic_val = max(self.base_weights.get('ic', 1.0) / time_factor, self.min_weight)

        self.weights['pde'] = float(pde_val)
        self.weights['bc'] = float(bc_val)
        self.weights['ic'] = float(ic_val)
        self._normalize_weights()

        changed = any(abs(self.weights[k] - old_weights.get(k, 0.0)) > 1e-9 for k in self.weights)
        self.error_history.append({
            'epoch': epoch,
            'high_freq_ratio': high_ratio,
            'weights': self.weights.copy(),
            'losses': current_losses.copy(),
            'changed': changed
        })

        # if changed:
            # print(f"[AdaptiveLossWeighter] epoch={epoch} hf_ratio={high_ratio:.4f} weights={self.weights}")

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                # normalize to sum to 3 (previously sum to 1)
                self.weights[key] = self.weights[key] / total * 3.0
