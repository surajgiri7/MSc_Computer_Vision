import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------


def calculate_derivative(img):
    img_blur = cv2.medianBlur(img, 15)
    img_dx = cv2.Sobel(img_blur, -1, 1, 0, 7).astype('float32') / 255
    img_dy = cv2.Sobel(img_blur, -1, 0, 1, 7).astype('float32') / 255
    return img_dx, img_dy

def calculate_energy(V, img_dx, img_dy, alpha):
    dx_i = lambda px: img_dx[px[1],px[0]]
    dy_i = lambda px: img_dy[px[1],px[0]]
    ext_e = lambda px : -(dx_i(px) * 2 + dy_i(px) * 2)
    dist_avg = np.mean([np.linalg.norm(V[i]-V[(i+1) % len(V)]) for i in range(len(V))])
    int_e = lambda u,v : alpha * (np.linalg.norm(u-v) - dist_avg + 0.5) ** 2
    return ext_e, int_e

def dynamic_programming(V, ext_e, int_e):
    unary = []
    for v in V:
        u = [{'ext': ext_e([x,y]), 'pos': np.array([x,y]), 'parent': None, 'acc': 0} 
             for x_offset in range(-1,2) for y_offset in range(-1,2) 
             for x, y in [(v[0] + x_offset, v[1] + y_offset)]]
        unary.append(u)
    for i in range(1, len(V)):
        for u in unary[i]:
            v = min(unary[i-1], key=lambda v: v['acc'] + v['ext'] + int_e(u['pos'], v['pos']))
            u['acc'] = v['acc'] + v['ext'] + int_e(u['pos'], v['pos'])
            u['parent'] = v
    smallest_node = min(unary[-1], key=lambda v: v['acc'] + v['ext'])
    V_new = [smallest_node['pos']]
    while smallest_node['parent'] is not None:
        smallest_node = smallest_node['parent']
        V_new.insert(0, smallest_node['pos'])
    V_new.append(V_new.pop(0))
    return np.array(V_new[::-1])

def run(fpath, radius):
    img, V = load_data(fpath, radius)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200
    img_dx, img_dy = calculate_derivative(img)
    alpha = 2
    for t in range(n_steps):
        ext_e, int_e = calculate_energy(V, img_dx, img_dy, alpha)
        V = dynamic_programming(V, ext_e, int_e)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)
    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=130)