import torch
import torch.nn as nn
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

from SpectrallyAdaptive.adaptive_pinn import AdaptiveSpectralPINN
from dataget import BurgersDataGenerator, save_dataset, load_dataset
import os
from vis_eval import evaluate_model, plot_solution_comparison, plot_training_history
from SpectrallyAdaptive.fixed_spectral_PINN import FixedAdaptivePINN
from SpectrallyAdaptive.adaptive_loss import AdaptiveLossWeighter

def train_adaptive_pinn(data_dict, n_epochs=5000, lr=0.001):
    model = AdaptiveSpectralPINN(domain_bounds=[(-1, 1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training Adaptive Spectral PINN...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = model.compute_adaptive_loss(data_dict, epoch)
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            current_weights = model.loss_weighter.weights
            current_losses = {'total': total_loss.item(), 'pde': model.history['pde_loss'][-1], 'bc': model.history['bc_loss'][-1], 'ic': model.history['ic_loss'][-1]}
            high_freq_ratio = model.history['high_freq_ratios'][-1] if model.history['high_freq_ratios'] else 0.0
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}")
            print(f"  Weights - PDE: {current_weights['pde']:.3f}, BC: {current_weights['bc']:.3f}, IC: {current_weights['ic']:.3f}")
            print(f"  High-freq ratio: {high_freq_ratio:.3f}")

    return model


def train_fixed_pinn(data_dict, n_epochs=400, lr=0.0015):
    model = FixedAdaptivePINN(domain_bounds=[(-1, 1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("="*60)
    print("FIXED INCREMENTAL ADAPTATION (debug)")
    print("="*60)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = model.compute_adaptive_loss(data_dict, epoch)
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0 and epoch > 0:
            print(f"\nEpoch {epoch}: Total Loss = {total_loss.item():.6f}")
            print(f"Current weights: {model.weights}")
            print(f"Total adaptation events: {model.adaptation_count}")

    return model


def main():
    print("\n" + "="*50)
    print("STARTING ADAPTIVE FREQUENCY APPROACH")
    print("="*50)

    data_path = os.path.join('data', 'burgers_dataset.npz')

    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        train_data, test_data = load_dataset(data_path)
    else:
        print("Generating training and test data with exact Cole-Hopf solution...")
        data_gen = BurgersDataGenerator(nu=0.01, L=1.0, T=1.0)

        train_data = data_gen.generate_training_data(
            n_collocation=1000, 
            n_boundary=100, 
            n_initial=100
        )

        test_data = data_gen.generate_test_data(n_x=100, n_t=50)

        print(f"Saving generated dataset to {data_path}...")
        save_dataset(data_path, train_data, test_data)

    adaptive_model = train_adaptive_pinn(train_data, n_epochs=5000, lr=0.001)

    print("\nEvaluating adaptive model...")
    adaptive_predictions, adaptive_mse, adaptive_rel_l2 = evaluate_model(adaptive_model, test_data)

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Baseline PINN:")
    print(f"  Test MSE: {0.457547:.6f}")
    print(f"  Relative L2 Error: {1.134239:.6f}")
    print(f"Adaptive Spectral PINN:")
    print(f"  Test MSE: {adaptive_mse:.6f}")
    print(f"  Relative L2 Error: {adaptive_rel_l2:.6f}")
    print(f"Improvement: {((1.134239 - adaptive_rel_l2) / 1.134239 * 100):.1f}%")

    plot_training_history(adaptive_model.history, "Adaptive Spectral PINN Training History")
    plot_solution_comparison(adaptive_model, test_data, "Adaptive Spectral PINN Solution")

    plt.figure(figsize=(10, 4))
    weights_history = adaptive_model.history['weights']
    epochs = range(len(weights_history))
    pde_weights = [w['pde'] for w in weights_history]
    bc_weights = [w['bc'] for w in weights_history] 
    ic_weights = [w['ic'] for w in weights_history]

    plt.plot(epochs, pde_weights, label='PDE Weight', linewidth=2)
    plt.plot(epochs, bc_weights, label='BC Weight', linewidth=2)
    plt.plot(epochs, ic_weights, label='IC Weight', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Adaptive Loss Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    main()
