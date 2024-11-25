import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

results_directory = "results"
os.makedirs(results_directory, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)  # Ensure consistent weight initialization
        self.lr = lr
        self.activation_fn = activation
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim)
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim)
        self.bias_output = np.zeros((1, output_dim))

    def activation(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, Z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(Z) ** 2
        elif self.activation_fn == 'relu':
            return np.where(Z > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # Input to hidden
        self.Z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.A1 = self.activation(self.Z1)
        # Hidden to output
        self.Z2 = np.dot(self.A1, self.weights_hidden_output) + self.bias_output
        self.A2 = np.tanh(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = y.shape[0]

        # Calculate derivatives
        delta_output = (self.A2 - y) / m
        delta_Z2 = delta_output * (1 - self.A2 ** 2)
        delta_W2 = np.dot(self.A1.T, delta_Z2)
        delta_b2 = np.sum(delta_Z2, axis=0, keepdims=True)

        delta_A1 = np.dot(delta_Z2, self.weights_hidden_output.T)
        delta_Z1 = delta_A1 * self.activation_derivative(self.Z1)
        delta_W1 = np.dot(X.T, delta_Z1)
        delta_b1 = np.sum(delta_Z1, axis=0, keepdims=True)

        # Save gradients for visualization
        self.delta_W1 = delta_W1
        self.delta_W2 = delta_W2

        # Update weights and biases with gradient descent
        self.weights_input_hidden -= self.lr * delta_W1
        self.bias_hidden -= self.lr * delta_b1
        self.weights_hidden_output -= self.lr * delta_W2
        self.bias_output -= self.lr * delta_b2

def generate_data(n_samples=100):
    np.random.seed(0)
    inputs = np.random.randn(n_samples, 2)
    targets = (inputs[:, 0] ** 2 + inputs[:, 1] ** 2 > 1).astype(int) * 2 - 1
    targets = targets.reshape(-1, 1)
    return inputs, targets

def update(frame_number, mlp, ax_hidden, ax_input, ax_gradient, X, y):
    # Clear all axes
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Plot hidden features with distinguishable colors
    hidden_activations = mlp.A1
    ax_hidden.scatter(
        hidden_activations[:, 0],
        hidden_activations[:, 1],
        hidden_activations[:, 2],
        c=y.ravel(),
        cmap='bwr',  # Colormap for better distinction
        alpha=0.7
    )
    
    # Plot decision hyperplane in hidden space
    weight_vector = mlp.weights_hidden_output[:, 0]
    bias_value = mlp.bias_output[0, 0]
    x_vals = np.linspace(hidden_activations[:, 0].min(), hidden_activations[:, 0].max(), 12)
    y_vals = np.linspace(hidden_activations[:, 1].min(), hidden_activations[:, 1].max(), 12)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = (-weight_vector[0] * X_grid - weight_vector[1] * Y_grid - bias_value) / weight_vector[2]
    ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='cyan')
    
    # Plot distorted input space transformed by the hidden layer
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    hidden_grid = mlp.activation(np.dot(grid_points, mlp.weights_input_hidden) + mlp.bias_hidden)
    ax_hidden.plot_surface(
        hidden_grid[:, 0].reshape(xx.shape),
        hidden_grid[:, 1].reshape(xx.shape),
        hidden_grid[:, 2].reshape(xx.shape),
        color='yellow',
        alpha=0.3
    )
    
    # Plot input layer decision boundary
    output_grid = mlp.forward(grid_points).reshape(xx.shape)
    ax_input.contourf(xx, yy, output_grid, levels=[-1, 0, 1], alpha=0.3, colors=['green', 'blue'])
    ax_input.contour(xx, yy, output_grid, levels=[0], colors='black')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
        
    # Set titles and labels with appropriate fontsize
    ax_input.set_title(f'Input Space Decision Boundary at Step {frame_number}', fontsize=13)
    ax_input.set_xlabel('Feature 1', fontsize=11)
    ax_input.set_ylabel('Feature 2', fontsize=11)
    
    # Define neuron positions and labels
    layer_x_positions = [0.15, 0.5, 0.85]
    nodes_per_layer = [2, 3, 1]
    positions = {
        (layer, node): (
            layer_x_positions[layer],
            np.linspace(0, 1, nodes_per_layer[layer])[node]
        )
        for layer in range(3)
        for node in range(nodes_per_layer[layer])
    }
    
    node_labels = {
        (layer, node): f'$x_{{{node+1}}}$' if layer == 0 else
                       (f'$h_{{{node+1}}}$' if layer == 1 else '$y$')
        for layer in range(3)
        for node in range(nodes_per_layer[layer])
    }
    
    # Plot neurons and labels
    for (layer, node), (x_pos, y_pos) in positions.items():
        ax_gradient.add_patch(Circle((x_pos, y_pos), 0.03, color='purple', zorder=-1))
        ax_gradient.text(
            x_pos,
            y_pos + 0.05,
            node_labels[(layer, node)],
            ha='center',
            fontsize=9,
            zorder=3
        )
    
    # Plot edges with gradient-based linewidth
    scale_factor = 120
    
    for i in range(nodes_per_layer[0]):
        for j in range(nodes_per_layer[1]):
            lw = abs(mlp.delta_W1[i, j]) * scale_factor
            ax_gradient.plot(
                [positions[(0, i)][0], positions[(1, j)][0]],
                [positions[(0, i)][1], positions[(1, j)][1]],
                'k-',
                linewidth=lw,
                zorder=-2
            )
    
    for j in range(nodes_per_layer[1]):
        lw = abs(mlp.delta_W2[j, 0]) * scale_factor
        ax_gradient.plot(
            [positions[(1, j)][0], positions[(2, 0)][0]],
            [positions[(1, j)][1], positions[(2, 0)][1]],
            'k-',
            linewidth=lw,
            zorder=-2
        )
    
    # Set gradient view titles and labels with appropriate fontsize
    ax_gradient.set_title(f'Network Gradients at Step {frame_number}', fontsize=13)
    ax_gradient.set_xlabel('Horizontal Position', fontsize=11)
    ax_gradient.set_ylabel('Vertical Position', fontsize=11)
    
    ax_gradient.set_xlim(-0.15, 1.15)
    ax_gradient.set_ylim(-0.15, 1.15)
    ax_gradient.set_aspect('equal')
    ax_gradient.set_xticks(np.linspace(0, 1, 5))
    ax_gradient.set_yticks(np.linspace(0, 1, 5))
    
    ax_gradient.grid(True, linestyle='--', linewidth=0.6, alpha=0.7, zorder=2)
    
    ax_gradient.tick_params(axis='both', which='major', labelsize=7)

def visualize(activation, learning_rate, total_steps):
    inputs, targets = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=learning_rate, activation=activation)
    
    matplotlib.use('agg')
    fig = plt.figure(figsize=(18, 6))
    
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    
    ani = FuncAnimation(
        fig,
        partial(
            update,
            mlp=mlp,
            ax_hidden=ax_hidden,
            ax_input=ax_input,
            ax_gradient=ax_gradient,
            X=inputs,
            y=targets
        ),
        frames=total_steps//8,
        repeat=False
    )
    
    ani.save(
        os.path.join(results_directory, "visualize.gif"),
        writer='pillow',
        fps=12,
        dpi=120
    )
    
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
