import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
import torch.nn as nn

class ExactBurgersSolution:
    """Exact solution to Burgers' equation using Cole-Hopf transformation"""
    
    def __init__(self, nu: float = 0.01):
        self.nu = nu
    
    def _F(self, y, x, t):
        """Helper function for the exact solution"""
        return np.exp(-(np.cos(np.pi * y) / (2 * np.pi * self.nu) + 
                       (x - y)**2 / (4 * self.nu * t)))
    
    def exact_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute exact solution using Cole-Hopf transformation"""
        
        def integrand(y, x_val, t_val):
            if t_val == 0:
                # Initial condition
                return -np.sin(np.pi * x_val)
            else:
                # Cole-Hopf solution
                F_val = self._F(y, x_val, t_val)
                numerator = np.sin(np.pi * y) * F_val
                denominator = F_val
                return numerator, denominator
        
        x = np.array(x, dtype=float)
        t = np.array(t, dtype=float)
        
        if x.ndim == 0:
            x = x.reshape(1)
        if t.ndim == 0:
            t = t.reshape(1)
            
        X, T = np.meshgrid(x, t, indexing='ij')
        U = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(t)):
                if T[i, j] == 0:
                    # Initial condition
                    U[i, j] = -np.sin(np.pi * X[i, j])
                else:
                    try:
                        # Numerical integration for numerator and denominator
                        num_integral, _ = quad(lambda y: integrand(y, X[i, j], T[i, j])[0], 
                                             -np.inf, np.inf, limit=100, epsabs=1e-6)
                        den_integral, _ = quad(lambda y: integrand(y, X[i, j], T[i, j])[1], 
                                             -np.inf, np.inf, limit=100, epsabs=1e-6)
                        
                        if den_integral != 0:
                            # Cole-Hopf gives u = - (numerator / denominator) for this setup
                            # initial condition is -sin(pi x), so the ratio must be negated
                            U[i, j] = - num_integral / den_integral
                        else:
                            U[i, j] = 0.0
                    except:
                        # Fallback for integration issues
                        U[i, j] = self._approximate_fallback(X[i, j], T[i, j])
        
        return U
    
    def _approximate_fallback(self, x, t):
        """Approximate solution when numerical integration fails"""
        # This is a more accurate approximation than the previous one
        # Based on asymptotic behavior for small nu
        if t < 0.1:
            return -np.sin(np.pi * x) * np.exp(-self.nu * np.pi**2 * t)
        else:
            # For larger times, use a shock profile approximation
            # The shock forms at x=0 and spreads with width ~ sqrt(nu*t)
            shock_width = np.sqrt(2 * self.nu * t)
            return -np.tanh(np.pi * x / (2 * shock_width))

class BurgersDataGenerator:
    """Generates exact solution and training data for Burgers' equation"""
    
    def __init__(self, nu: float = 0.01, L: float = 1.0, T: float = 1.0):
        self.nu = nu  # viscosity
        self.L = L    # domain length [-L, L]
        self.T = T    # final time
        self.exact_solver = ExactBurgersSolution(nu)
    
    def exact_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute exact solution using Cole-Hopf transformation"""
        return self.exact_solver.exact_solution(x, t)
    
    def generate_training_data(self, n_collocation: int, n_boundary: int, n_initial: int):
        """Generate training points for PINN"""
        
        # Collocation points (inside the domain)
        x_colloc = np.random.uniform(-self.L, self.L, n_collocation)
        t_colloc = np.random.uniform(0, self.T, n_collocation)
        colloc_points = np.column_stack([x_colloc, t_colloc])
        
        # Boundary points
        # Left boundary (x = -L)
        t_left = np.random.uniform(0, self.T, n_boundary // 2)
        x_left = -self.L * np.ones_like(t_left)
        # Right boundary (x = L)  
        t_right = np.random.uniform(0, self.T, n_boundary // 2)
        x_right = self.L * np.ones_like(t_right)
        
        boundary_points = np.column_stack([
            np.concatenate([x_left, x_right]),
            np.concatenate([t_left, t_right])
        ])
        boundary_values = np.zeros(2 * (n_boundary // 2))  # u=0 at boundaries
        
        # Initial condition points (t = 0)
        x_initial = np.random.uniform(-self.L, self.L, n_initial)
        t_initial = np.zeros(n_initial)
        initial_points = np.column_stack([x_initial, t_initial])
        initial_values = -np.sin(np.pi * x_initial)  # u(x,0) = -sin(πx)
        
        # Convert to tensors
        colloc_tensor = torch.tensor(colloc_points, dtype=torch.float32, requires_grad=True)
        boundary_tensor = torch.tensor(boundary_points, dtype=torch.float32)
        boundary_vals_tensor = torch.tensor(boundary_values, dtype=torch.float32)
        initial_tensor = torch.tensor(initial_points, dtype=torch.float32)  
        initial_vals_tensor = torch.tensor(initial_values, dtype=torch.float32)
        
        return {
            'collocation': colloc_tensor,
            'boundary_points': boundary_tensor,
            'boundary_values': boundary_vals_tensor, 
            'initial_points': initial_tensor,
            'initial_values': initial_vals_tensor
        }
    
    def generate_test_data(self, n_x: int = 100, n_t: int = 50):
        """Generate grid of test points for validation"""
        x_test = np.linspace(-self.L, self.L, n_x)
        t_test = np.linspace(0, self.T, n_t)
        X, T = np.meshgrid(x_test, t_test, indexing='ij')
        
        print("Computing exact solution... (this may take a moment)")
        U_exact = self.exact_solution(x_test, t_test)
        
        test_points = np.column_stack([X.flatten(), T.flatten()])
        test_values = U_exact.flatten()
        
        return {
            'points': torch.tensor(test_points, dtype=torch.float32),
            'values': torch.tensor(test_values, dtype=torch.float32),
            'grid': (X, T, U_exact)
        }

# Let's visualize the exact solution to verify it's working correctly
def plot_exact_solution():
    """Plot the exact Burgers' equation solution"""
    data_gen = BurgersDataGenerator(nu=0.01, L=1.0, T=1.0)
    test_data = data_gen.generate_test_data(n_x=200, n_t=100)
    
    X, T, U_exact = test_data['grid']
    
    plt.figure(figsize=(12, 4))
    
    # 2D contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contourf(T, X, U_exact, levels=50, cmap='jet')
    plt.colorbar(contour)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title('Exact Burgers Solution (Cole-Hopf)')
    
    # 1D slices at different times
    plt.subplot(1, 2, 2)
    times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    for i, t in enumerate(times):
        t_idx = int(t * (U_exact.shape[1] - 1))
        plt.plot(X[:, t_idx], U_exact[:, t_idx], color=colors[i], label=f't={t}', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Solution Profiles at Different Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return test_data


def save_dataset(path, train_data: dict, test_data: dict):
    """Save training and test data to an .npz file.

    Arrays are converted to numpy before saving. Expects the same keys as
    generate_training_data / generate_test_data return values.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def to_numpy(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    np.savez_compressed(
        path,
        collocation=to_numpy(train_data['collocation']),
        boundary_points=to_numpy(train_data['boundary_points']),
        boundary_values=to_numpy(train_data['boundary_values']),
        initial_points=to_numpy(train_data['initial_points']),
        initial_values=to_numpy(train_data['initial_values']),
        test_points=to_numpy(test_data['points']),
        test_values=to_numpy(test_data['values']),
        X=test_data['grid'][0],
        T=test_data['grid'][1],
        U_exact=test_data['grid'][2]
    )


def load_dataset(path):
    """Load dataset saved with save_dataset and return (train_data, test_data) as dicts with torch tensors."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.load(path)

    train = {
        'collocation': torch.tensor(data['collocation'], dtype=torch.float32, requires_grad=True),
        'boundary_points': torch.tensor(data['boundary_points'], dtype=torch.float32),
        'boundary_values': torch.tensor(data['boundary_values'], dtype=torch.float32),
        'initial_points': torch.tensor(data['initial_points'], dtype=torch.float32),
        'initial_values': torch.tensor(data['initial_values'], dtype=torch.float32),
    }

    test = {
        'points': torch.tensor(data['test_points'], dtype=torch.float32),
        'values': torch.tensor(data['test_values'], dtype=torch.float32),
        'grid': (data['X'], data['T'], data['U_exact'])
    }

    return train, test