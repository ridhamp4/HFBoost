import matplotlib.pyplot as plt
import os
import torch
import numpy as np 

def plot_training_history(history, title="Training History", save_path=None):
    """Plot training loss history. If save_path is provided, save the figure and don't show interactively."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(history['total_loss'], label='Total Loss', linewidth=2)
    plt.semilogy(history['pde_loss'], label='PDE Loss', alpha=0.7)
    plt.semilogy(history['bc_loss'], label='BC Loss', alpha=0.7)
    plt.semilogy(history['ic_loss'], label='IC Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, test_data):
    """Evaluate model on test data"""
    with torch.no_grad():
        predictions = model(test_data['points'])
        exact = test_data['values']
        mse = torch.mean((predictions.squeeze() - exact)**2)
        relative_l2 = torch.norm(predictions.squeeze() - exact) / torch.norm(exact)
        
    print(f"Test MSE: {mse.item():.6f}")
    print(f"Relative L2 Error: {relative_l2.item():.6f}")
    
    return predictions, mse.item(), relative_l2.item()

def plot_solution_comparison(model, test_data, title="Solution Comparison", save_path=None):
    """Plot model prediction vs exact solution"""
    X, T, U_exact = test_data['grid']
    
    with torch.no_grad():
        test_points = test_data['points']
        U_pred = model(test_points).numpy().reshape(X.shape)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Exact solution
    im1 = axes[0].contourf(T, X, U_exact, levels=50, cmap='jet')
    axes[0].set_title('Exact Solution')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Space')
    plt.colorbar(im1, ax=axes[0])
    
    # Predicted solution
    im2 = axes[1].contourf(T, X, U_pred, levels=50, cmap='jet') 
    axes[1].set_title('Predicted Solution')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Space')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(U_exact - U_pred)
    im3 = axes[2].contourf(T, X, error, levels=50, cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Time') 
    axes[2].set_ylabel('Space')
    plt.colorbar(im3, ax=axes[2])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.tight_layout()
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def contour_plot_fields(T, X, U_exact, U_pred, outpath: str = None, titles=None, cmap='jet'):
    """Reusable contour plot for exact, predicted and error fields.

    T, X should be meshgrid arrays matching U_exact / U_pred shapes.
    """
    if titles is None:
        titles = ['Exact Solution', 'Predicted Solution', 'Absolute Error']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].contourf(T, X, U_exact, levels=50, cmap=cmap)
    axes[0].set_title(titles[0])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Space')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(T, X, U_pred, levels=50, cmap=cmap)
    axes[1].set_title(titles[1])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Space')
    plt.colorbar(im2, ax=axes[1])

    error = np.abs(U_exact - U_pred)
    im3 = axes[2].contourf(T, X, error, levels=50, cmap='hot')
    axes[2].set_title(titles[2])
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Space')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath)
        plt.close(fig)
    else:
        plt.show()


def contour_plot_models(model, test_data, outpath: str = None, titles=None, cmap='jet'):
    """Wrapper that evaluates `model` on `test_data` and calls `contour_plot_fields`.

    Expects test_data to contain 'grid' (X, T, U_exact) and 'points'.
    """
    X, T, U_exact = test_data['grid']
    with torch.no_grad():
        U_pred = model(test_data['points']).numpy().reshape(X.shape)

    contour_plot_fields(T, X, U_exact, U_pred, outpath=outpath, titles=titles, cmap=cmap)