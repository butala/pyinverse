import numpy as np

"""
The best reference for this is Jeff Fessler's course notes: https://web.eecs.umich.edu/~fessler/book/

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


def rect_conv_rect(x, a=1, b=1):
    """Scaled rect convovled wtih scaled rect (CHECK IF THIS DUPLICATES srect_conv_srect)."""
    assert a > 0
    assert b > 0
    return step1(x + 1/(2*a) + 1/(2*b)) - step1(x - 1/(2*a) + 1/(2*b)) - step1(x + 1/(2*a) - 1/(2*b)) + step1(x - 1/(2*a) - 1/(2*b))


def step(x):
    """Heaviside step function u(x)."""
    y = np.zeros_like(x)
    y[x > 0] = 1
    return y


def step1(x):
    """Convolution of step functions."""
    y = np.zeros_like(x)
    y[x > 0] = x[x > 0]
    return y


def step2(x):
    """Convolution of three step functions."""
    y = np.zeros_like(x)
    y[x > 0] = 1/2 * x[x > 0]**2
    return y


def tri(x, b=1):
    """Triangle function tri(bx) where tri(x) = rect(x) * rect(x)."""
    assert b > 0
    return b*step1(x + 1/b) - 2*b*step1(x) + b*step1(x - 1/b)


def rtri(x, a, b):
    """Convolution of rect(ax) with tri(bx)."""
    assert a > 0
    assert b > 0
    return b*(step2(x + 1/(2*a) + 1/b) - 2*step2(x + 1/(2*a)) + step2(x + 1/(2*a) - 1/b) - step2(x - 1/(2*a) + 1/b) + 2*step2(x - 1/(2*a)) - step2(x - 1/(2*a) - 1/b))


def radon_rect(theta, r, a, b):
    """Fessler (3.2.40)"""
    assert a > 0
    assert b > 0
    theta = theta % (2*np.pi)
    c_t = np.cos(theta)
    s_t = np.sin(theta)
    if np.allclose(abs(a*c_t), abs(b*s_t)):
        return np.hypot(a, b) * tri*(r/((a*b)/np.hypot(a, b)))
    elif abs(theta) in [0, np.pi]:
        return b * rect(r/a)
    elif abs(theta) in [np.pi/2, 3*np.pi/2]:
        return a * rect(r/b)
    else:
        d_max = (abs(a*c_t) + abs(b*s_t)) / 2
        d_break = abs(abs(a*c_t) - abs(b*s_t)) / 2
        l_max = abs(a*b)/max(abs(a*c_t), abs(b*s_t))
        return 1/abs(c_t*s_t) * (d_max*tri(r/d_max) - d_break*tri(r/d_break))


def rect_conv_radon_rect(theta, r, a, b, alpha):
    """Based off of Fessler (3.2.40)"""
    assert a > 0
    assert b > 0
    assert alpha > 0
    theta %= 2*np.pi
    c_t = np.cos(theta)
    s_t = np.sin(theta)
    if np.allclose(abs(a*c_t), abs(b*s_t)):
        return np.hypot(a, b) * rtri(r, 1/alpha, 1/((a*b)/np.hypot(a, b)))
    elif abs(theta) in [0, np.pi]:
        return b * rect_conv_rect(r, 1/a, 1/alpha)
    elif abs(theta) in [np.pi/2, 3*np.pi/2]:
        return a * rect_conv_rect(r, 1/b, 1/alpha)
    else:
        d_max = (abs(a*c_t) + abs(b*s_t)) / 2
        d_break = abs(abs(a*c_t) - abs(b*s_t)) / 2
        l_max = abs(a*b)/max(abs(a*c_t), abs(b*s_t))
        return 1/abs(c_t*s_t) * (d_max*rtri(r, 1/alpha, 1/d_max) - d_break*rtri(r, 1/alpha, 1/d_break))
