import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
#from bokeh.palettes import


def k(xs, ys, sigma=1, l=1):
    """Sqared Exponential kernel as above but designed to return the whole
    covariance matrix - i.e. the pairwise covariance of the vectors xs & ys.
    Also with two parameters which are discussed at the end."""

    # Pairwise difference matrix.
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)

def m(x):
    """The mean function. As discussed, we can let the mean always be zero."""
    return np.zeros_like(x)


colors = ["blue","blueviolet","brown"]

def plot_unit_gaussian_samples(D):
    p = figure(plot_width=800, plot_height=500)
    xs = np.linspace(-5, 5, D)
    K = k(xs, xs)
    mu = m(xs)
    for color in colors:
        ys = np.random.multivariate_normal(mu,K)
        p.circle(xs, ys, size=3, color=color)
        p.line(xs, ys, line_width=1, color=color)

    show(p)
#plot_unit_gaussian_samples(100)
# coefs[i] is the coefficient of x^i
coefs = [6, -2.5, -2.4, -0.1, 0.2, 0.03]

def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
       total += coef * (x ** exp)
    return total

x_obs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
y_obs = f(x_obs)

x_s = np.linspace(-8, 7, 80)
K = k(x_obs, x_obs)
K_s = k(x_obs, x_s)
K_ss = k(x_s, x_s)

K_sTKinv = np.matmul(K_s.T, np.linalg.pinv(K))

mu_s = m(x_s) + np.matmul(K_sTKinv, y_obs - m(x_obs))
Sigma_s = K_ss - np.matmul(K_sTKinv, K_s)
p = figure(plot_width=800, plot_height=600, y_range=(-7, 8))

y_true = f(x_s)
p.line(x_s, y_true, line_width=3, color='black', alpha=0.4,
      line_dash='dashed', legend='True f(x)')

p.cross(x_obs, y_obs, size=20, legend='Training data')

stds = np.sqrt(Sigma_s.diagonal())
err_xs = np.concatenate((x_s, np.flip(x_s, 0)))
err_ys = np.concatenate((mu_s + 2 * stds, np.flip(mu_s - 2 * stds, 0)))
p.patch(err_xs, err_ys, alpha=0.2, line_width=0, color='grey',
       legend='Uncertainty')

for color in colors:
   y_s = np.random.multivariate_normal(mu_s, Sigma_s)
   p.line(x_s, y_s, line_width=1, color=color)

p.line(x_s, mu_s, line_width=3, color='blue', alpha=0.4, legend='Mean')
show(p)