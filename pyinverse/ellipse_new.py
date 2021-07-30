import numpy as np

from .radon_new import radon_translate, radon_affine_scale


def integral_sqrt_a2_minus_x2(x, a):
    """ ??? """
    return 0.5*x*np.sqrt(a**2 - x**2) + 0.5*a**2*np.arctan2(x, np.sqrt(a**2 - x**2))


def proj_ellipse_rect(e, theta_rad, t, a):
    """ ??? """
    theta_prime = theta_rad - e.phi_rad
    theta_prime2 = e.phi_rad + theta_rad
    t_prime = radon_translate(theta_rad, t, e.x0, e.y0)
    theta_prime2, t_prime2, scale_factor = radon_affine_scale(theta_prime, t_prime, 1/e.a, 1/e.b)
    a_prime = a / scale_factor * e.a * e.b
    y = np.zeros_like(t_prime2)

    I = np.abs(t_prime2) < 1 + 1/(2*a_prime)
    t_prime2_left = t_prime2[I] - 1/(2*a_prime)
    t_prime2_left[t_prime2_left < -1] = -1
    t_prime2_right = t_prime2[I] + 1/(2*a_prime)
    t_prime2_right[t_prime2_right > 1] = 1

    I1 = integral_sqrt_a2_minus_x2(t_prime2_right, 1)
    I2 = integral_sqrt_a2_minus_x2(t_prime2_left, 1)

    y[I] = 2*e.rho*scale_factor*a_prime*(I1 - I2)
    return y
