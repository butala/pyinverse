import numpy as np

"""
The best reference for this is Jeff Fessler's course notes: https://web.eecs.umich.edu/~fessler/course/516/l/c-tomo.pdf

This is from a course he taught: https://web.eecs.umich.edu/~fessler/course/516/
"""


def rect(t):
    """Rectangle function."""
    f = np.zeros_like(t)
    I = np.abs(t) < 0.5
    f[I] = 1
    f[np.abs(t) == 0.5] = 0.5
    return f


def srect(t, a):
    """Scaled rectangle function."""
    return rect(a*t)


def srect_conv_srect(t, a, b):
    """Scaled rectangle convolved with scaled rectangle."""
    assert a > 0 and b > 0
    if a < b:
        return srect_conv_srect(t, b, a)
    f = np.zeros_like(t)
    I1 = np.abs(t) < (a+b)/(2*a*b)
    I2 = np.abs(t) > (a-b)/(2*a*b)
    I = I1 & I2
    f[I] = (a+b)/(2*a*b) - np.abs(t[I])
    f[~I2] = 1/a
    return f


def srect_2D_proj(theta, t, a, b):
    """Projection of the scaled rectangle function."""
    theta = np.asarray(theta)
    if a < b:
        return srect_2D_proj(theta - np.pi/2, t, b, a)
    P = np.empty((len(t), len(theta)))
    for k, theta_k in enumerate(theta % (2*np.pi)):
        if theta_k == 0:
            p = srect(t, a) / b
        elif theta_k == np.pi/2:
            p = srect(t, b) / a
        elif theta_k == np.pi:
            p = srect(-t, a) / b
        elif theta_k == 3*np.pi /2:
            p = srect(-t, b) / a
        else:
            if theta_k < np.pi/2:
                sign = 1
            elif theta_k < np.pi:
                sign = -1
            elif theta_k < 3*np.pi / 2:
                sign = 1
            else:
                sign = -1
            abs_cos = np.abs(np.cos(theta_k))
            abs_sin = np.abs(np.sin(theta_k))
            p = 1/(abs_cos * abs_sin) * srect_conv_srect(t, a/abs_cos, b/abs_sin)
        P[:, k] = p
    return P


def srect_2D_proj_ramp(theta, t, a, b):
    """Ramp filtered projection of the scaled rectangle function."""
    theta = np.asarray(theta)
    # we use the opposite rect scaling convention of Fessler in his notes
    a = 1/a
    b = 1/b
    #a, b = b, a
    #if a < b:
    #    return srect_2D_proj_ramp(theta - np.pi/2, t, b, a)
    P = np.empty((len(t), len(theta)))
    for k, theta_k in enumerate(theta % (2*np.pi)):
        if theta_k == 0 or theta_k == np.pi:
            p = -2*a*b/np.pi**2 / (4*t**2 - a**2)
        elif theta_k == np.pi/2 or theta_k == 3*np.pi/2:
            p = -2*a*b/np.pi**2 / (4*t**2 - b**2)
        else:
            p = 1/(2*np.pi**2*np.cos(theta_k)*np.sin(theta_k)) * np.log(np.abs((t**2 - ((a*np.cos(theta_k)+b*np.sin(theta_k))/2)**2)/(t**2 - ((a*np.cos(theta_k)-b*np.sin(theta_k))/2)**2)))
        P[:, k] = p
    return P
