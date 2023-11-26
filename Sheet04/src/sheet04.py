import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.optimize import minimize

def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    V_plt = np.append(V.reshape(-1), V[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(V[:, 0], V[:, 1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))

def load_data(fpath, radius):
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 100  # Increase the number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V

def calculate_curvature(snake):
    dx = np.gradient(snake[:, 0])
    dy = np.gradient(snake[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / np.sqrt((dx**2 + dy**2)**3)
    return 0.5 * np.sum(curvature)

def calculate_image_gradient(snake, Im):
    grad_x = cv2.Sobel(Im, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(Im, cv2.CV_64F, 0, 1, ksize=3)

    snake_coords = np.clip(snake.astype(int), [0, 0], [Im.shape[1] - 1, Im.shape[0] - 1])

    grad_at_snake = np.sqrt(grad_y[snake_coords[:, 1], snake_coords[:, 0]]**2 +
                            grad_x[snake_coords[:, 1], snake_coords[:, 0]]**2)

    return -np.sum(grad_at_snake)

def compute_energy(snake, Im, alpha, beta):
    internal = alpha * calculate_curvature(snake)
    external = beta * calculate_image_gradient(snake, Im)
    total_energy = internal + external
    return total_energy

def objective_function(x, Im, alpha, beta):
    snake = x.reshape((-1, 2))
    return compute_energy(snake, Im, alpha, beta)

def optimize_parameters(fpath, radius, alpha_range, beta_range):
    Im, initial_snake = load_data(fpath, radius)

    best_params = None
    best_energy = np.inf

    for alpha in alpha_range:
        for beta in beta_range:
            # Flatten the initial snake points for optimization
            initial_guess = initial_snake.flatten()

            # Define the bounds for optimization
            bounds = [(0, Im.shape[1]) for _ in range(len(initial_guess))]

            # Use L-BFGS-B method for optimization
            result = minimize(objective_function, initial_guess, args=(Im, alpha, beta),
                              method='L-BFGS-B', bounds=bounds)

            # Reshape the result to get the optimized snake
            optimal_snake = result.x.reshape((-1, 2))

            # Evaluate the energy of the optimized snake
            energy = compute_energy(optimal_snake, Im, alpha, beta)

            # Update the best parameters if the current configuration has lower energy
            if energy < best_energy:
                best_energy = energy
                best_params = {'alpha': alpha, 'beta': beta, 'snake': optimal_snake}

    return best_params

def run_with_optimal_parameters(fpath, radius, optimal_params):
    Im, initial_snake = load_data(fpath, radius)

    # Use the optimal parameters
    alpha = optimal_params['alpha']
    beta = optimal_params['beta']

    # Flatten the initial snake points for optimization
    initial_guess = initial_snake.flatten()

    # Define the bounds for optimization
    bounds = [(0, Im.shape[1]) for _ in range(len(initial_guess))]

    # Use L-BFGS-B method for optimization
    result = minimize(objective_function, initial_guess, args=(Im, alpha, beta),
                      method='L-BFGS-B', bounds=bounds)

    # Reshape the result to get the optimized snake
    optimal_snake = result.x.reshape((-1, 2))

    # Visualize the result
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(Im, cmap='gray')
    plot_snake(ax, optimal_snake)
    plt.show()

if __name__ == '__main__':
    # Define the ranges for alpha and beta
    alpha_range = np.linspace(0.01, 0.1, 5)
    beta_range = np.linspace(0.01, 0.1, 5)

    # Perform parameter optimization
    optimal_params = optimize_parameters('../images/ball.png', radius=120, alpha_range=alpha_range, beta_range=beta_range)

    # Run with the optimal parameters
    run_with_optimal_parameters('./images/ball.png',120, optimal_params)
