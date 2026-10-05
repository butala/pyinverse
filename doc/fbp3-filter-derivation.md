
# Table of Contents

1.  [Summary](#summary)
2.  [Setting and notation](#setting-and-notation)
3.  [The 2-D case: the ramp is a Jacobian](#the-2-d-case-the-ramp-is-a-jacobian)
4.  [The 3-D case](#the-3-d-case)
    1.  [Fourier slice theorem for a rotating 2-D detector](#fourier-slice-theorem-for-a-rotating-2-d-detector)
    2.  [Change of variables and the Jacobian](#change-of-variables-and-the-jacobian)
    3.  [The inversion formula](#the-inversion-formula)
5.  [Why the multiplier is flat along $f_v$: slice decoupling](#why-the-multiplier-is-flat-along-f_v-slice-decoupling)
6.  [The bug: an isotropic 2-D ramp, and why it produces saturation](#the-bug-an-isotropic-2-d-ramp-and-why-it-produces-saturation)
    1.  [Why this survived for so long](#why-this-survived-for-so-long)
7.  [Numerical evidence](#numerical-evidence)
8.  [The general case: tilted detector, $\theta_0 \neq 0$](#the-general-case-tilted-detector-theta_0-neq-0)
9.  [Relation to FDK and other geometries](#relation-to-fdk-and-other-geometries)
10. [Reproducing](#reproducing)
11. [Appendix A: the determinant](#appendix-a-the-determinant)
12. [Appendix B: the discrete algorithm as implemented](#appendix-b-the-discrete-algorithm-as-implemented)
13. [References](#references)



<a id="summary"></a>

# Summary

The 3-D filtered backprojection (FBP) routine in this package
(`pyinverse.fbp3`) follows the standard recipe for a parallel-beam
scanner with a **planar 2-D detector rotating about an axis**: take the
2-D Fourier transform of each 2-D projection, multiply by a filter
defined on the detector-frequency plane $(f_u, f_v)$, inverse
transform, and backproject into the volume. That structure is correct.

The **value of the filter was wrong** &mdash; this is the bug, now corrected
(§6). The code used the *isotropic 2-D ramp*

$$W_{\text{code}}(f_u,f_v) \;=\; \sqrt{f_u^{2}+f_v^{2}}\,, \qquad (1)$$

whereas the correct multiplier is the *one-dimensional ramp*

$$W_{\text{correct}}(f_u,f_v) \;=\; |f_u|\,, \qquad (2)$$

which varies along the detector axis transverse to the rotation axis and
is **flat along the other axis**, $f_v$.

This is counter-intuitive: the correct “2-D filter” for 3-D FBP looks
1-D. The reason is that the ramp is not a physical filter at all &mdash; it
is the **Jacobian of a change of variables** in frequency space. In 2-D
FBP the ramp $|f|$ is the Jacobian of polar coordinates
$(f,\varphi)$ in the 2-D frequency plane. In 3-D FBP the natural
coordinates are $(\phi, f_u, f_v)$, and their Jacobian is $|f_u|$.

> **Take-aways**
> 
> 1.  Yes &mdash; 2-D FFT of the 2-D projection, multiply by a 2-D multiplier,
>     2-D inverse FFT, backproject. That is the right algorithm.
> 2.  The 2-D multiplier must be $W(f_u,f_v)=|f_u|$, not
>     $\sqrt{f_u^2+f_v^2}$.
> 3.  Using the isotropic ramp imposes a frequency-dependent gain
>     $\sqrt{f_x^2+f_y^2+f_z^2}\,/\,\sqrt{f_x^2+f_y^2} \neq 1$ that is
>     independent of the number of projections. This is the source of the
>     observed *saturation* (reconstruction error that does not improve as
>     projections are added).
> 4.  Because $|f_u|$ is constant in $f_v$, the 3-D reconstruction is
>     *exactly* per-slice 2-D FBP. This was verified numerically to machine
>     precision ($\sim 10^{-14}$) &mdash; see §7.
> 5.  The same construction covers a **tilted** detector
>     ($\theta_0 \neq 0$, §8): the filter is still $|f_u|$, but the
>     inversion formula now carries an explicit $\cos\theta_0$. Unlike
>     the untilted case, that geometry is intrinsically *incomplete* &mdash; a
>     double cone of half-angle $|\theta_0|$ about the rotation axis is
>     never sampled, so the error plateaus rather than converges.


<a id="setting-and-notation"></a>

# Setting and notation

Let $f:\mathbb{R}^3\to\mathbb{R}$ be the object, with compact support.
We consider a parallel-beam geometry with a planar detector
parameterised by $(u,v)$, rotating about the $z$ axis by an angle
$\phi$. With the detector plane containing the rotation axis (the
*non-tilted* case, `theta0 = 0` in the code), the ray that passes
through the point $(x,y,z)$ and is labelled by the detector
coordinates $(u,v)$ satisfies

$$u = x\cos\phi + y\sin\phi, \qquad v = z . \qquad (3)$$

That is, $\hat e_1(\phi)=(\cos\phi,\sin\phi,0)$ is the transverse
in-plane direction and $\hat e_2=(0,0,1)=\hat z$ is the detector axis
parallel to the rotation axis.

The measured projection at angle $\phi$ is

$$P_\phi(u,v) = \iint_{\mathbb{R}^2} f(x,y,v)\,\delta\!\big(u - (x\cos\phi + y\sin\phi)\big)\,dx\,dy = \mathcal{R}\big[f(\cdot,\cdot,v)\big](u,\phi)\,, \qquad (4)$$

so each *detector row* (fixed $v=z$) is exactly the 2-D Radon
transform of the corresponding slice of the object. Denote the 3-D
Fourier transform

$$F(\boldsymbol{\xi}) = \int_{\mathbb{R}^3} f(\mathbf{x})\,e^{-i2\pi\,\boldsymbol{\xi}\cdot\mathbf{x}}\,d^3x , \qquad \mathbf{x}=(x,y,z),\; \boldsymbol{\xi}=(f_x,f_y,f_z), \qquad (5)$$

and the 2-D Fourier transform of a projection

$$\hat P_\phi(f_u,f_v) = \iint_{\mathbb{R}^2} P_\phi(u,v)\,e^{-i2\pi(f_u u + f_v v)}\,du\,dv . \qquad (6)$$

The frequency variable $f_u$ passes through the transverse plane
$(\hat e_1,\hat e_2)$-coordinates, and $f_v$ labels the
$z$-frequency. All frequencies are in cycles per unit length (Hz)
rather than rad/s; this is the convention used internally by the package
(see Appendix B).


<a id="the-2-d-case-the-ramp-is-a-jacobian"></a>

# The 2-D case: the ramp is a Jacobian

It is worth first recalling why a ramp appears in the 2-D problem,
because the 3-D answer is the same argument in one more dimension.

The inversion formula for the 2-D Radon transform is obtained by writing
the inverse Fourier transform of the object in *polar* coordinates in
the frequency plane. With
$f(x,y)=\iint F(f_x,f_y)e^{i2\pi(f_x x+f_y y)}\,df_x\,df_y$ and
$(f_x,f_y)=f(\cos\varphi,\sin\varphi)$, the integration measure
becomes

$$df_x\,df_y = |f|\;df\,d\varphi . \qquad (7)$$

The factor $|f|$ is a **Jacobian**, not a filter in any physical sense.
Using the Fourier slice theorem of the 2-D problem
($\hat P_\varphi(f) = F(f\cos\varphi, f\sin\varphi)$), the inversion
reads

$$f(x,y) = \int_0^\pi d\varphi \int_{-\infty}^{\infty} df\; \underbrace{|f|}_{\text{Jacobian}}\; \hat P_\varphi(f)\; e^{i2\pi f u}, \qquad u = x\cos\varphi + y\sin\varphi . \qquad (8)$$

That is all the 2-D FBP ramp is: the Jacobian of polar coordinates
$(f,\varphi)$ of the 2-D frequency plane. (`pyinverse.fbp.ramp_filter`
implements exactly $|f|$, along the sinogram axis.)


<a id="the-3-d-case"></a>

# The 3-D case


<a id="fourier-slice-theorem-for-a-rotating-2-d-detector"></a>

## Fourier slice theorem for a rotating 2-D detector

Apply the Fourier transform (6) to the projection (4). Writing
$P_\phi(u,v)=\iint f(x,y,v)\delta\big(u-(x\cos\phi+y\sin\phi)\big)\,dx\,dy$
and applying the sifting property in $u$ while integrating in $v=z$:

$$\hat P_\phi(f_u,f_v) = \iiint_{\mathbb{R}^3} f(x,y,z)\, e^{-i2\pi\big(f_u(x\cos\phi + y\sin\phi) + f_v z\big)}\,dx\,dy\,dz = F(f_u\cos\phi,\; f_u\sin\phi,\; f_v)\,. \qquad (9)$$

This is the **3-D Fourier slice theorem**: a single 2-D projection gives
the object's 3-D Fourier transform on the *plane*

$$\Pi_\phi = \big\{\,\boldsymbol{\xi} : \boldsymbol{\xi} = f_u\,\hat e_1(\phi) + f_v\,\hat e_2 ,\; (f_u,f_v)\in\mathbb{R}^2 \,\big\}, \qquad (10)$$

i.e. the plane spanned by $\hat e_1(\phi)$ and $\hat e_2 = \hat z$,
which is the plane through the $f_z$ axis whose normal is
$\hat n(\phi) = \hat e_1(\phi)\times\hat e_2 = (\sin\phi,-\cos\phi,0)$.
As $\phi$ sweeps $[0,\pi)$ these planes rotate about the $f_z$
axis and fill all of frequency space (up to the measure-zero $f_z$
axis itself, which is covered by the limit $f_u\to 0$).


<a id="change-of-variables-and-the-jacobian"></a>

## Change of variables and the Jacobian

Now parametrise 3-D frequency space by $(\phi,f_u,f_v)$:
$\boldsymbol{\xi} = f_u\hat e_1(\phi) + f_v\hat e_2$, with
$\phi\in[0,\pi)$, $f_u\in\mathbb{R}$, $f_v\in\mathbb{R}$.
(Restricting $\phi$ to a half-turn removes the double cover: the map
$(\phi+\pi,-f_u,f_v)$ reproduces the same $\boldsymbol{\xi}$, so the
half-turn together with the whole real line for $f_u$ is a bijection
onto $\mathbb{R}^3\!\setminus\!\{f_z\text{-axis}\}$.)

The Jacobian matrix $\partial(f_x,f_y,f_z)/\partial(\phi,f_u,f_v)$ has
columns

$$\frac{\partial\boldsymbol{\xi}}{\partial\phi} = f_u\!\begin{pmatrix}-\sin\phi\\ \cos\phi\\ 0\end{pmatrix} + f_v\!\begin{pmatrix}\sin\phi\sin\theta\\ -\cos\phi\sin\theta\\ \cos\theta\end{pmatrix}, \qquad \frac{\partial\boldsymbol{\xi}}{\partial f_u} = \hat e_1, \qquad \frac{\partial\boldsymbol{\xi}}{\partial f_v} = \hat e_2 ,$$

evaluated here at $\theta=0$ (the non-tilted case), where
$\hat e_1=(\cos\phi,\sin\phi,0)$ and $\hat e_2=(0,0,1)$. Taking the
determinant (see Appendix A for the full computation, and §8 for general
tilt $\theta$):

$$\det\frac{\partial(f_x,f_y,f_z)}{\partial(\phi,f_u,f_v)} = -f_u , \qquad\text{so}\qquad d^3\xi = |f_u|\;d\phi\,df_u\,df_v . \qquad (11)$$

**This is the whole story.** The Jacobian is $|f_u|$ &mdash;
one-dimensional, with no $f_v$ dependence.


<a id="the-inversion-formula"></a>

## The inversion formula

Substituting (9) and (11) into the inverse Fourier transform
$f(\mathbf{x})=\int F(\boldsymbol{\xi})e^{i2\pi\boldsymbol{\xi}\cdot\mathbf{x}}\,d^3\xi$:

$$f(x,y,z) = \int_0^\pi d\phi \int_{-\infty}^{\infty}\!\! df_u \int_{-\infty}^{\infty}\!\! df_v\; \underbrace{|f_u|}_{\text{Jacobian}}\; \hat P_\phi(f_u,f_v)\; e^{i2\pi (f_u u + f_v v)}, \qquad \begin{aligned} u &= x\cos\phi + y\sin\phi,\\ v &= z . \end{aligned} \qquad (12)$$

Define the **filtered projection**

$$g_\phi(u,v) = \iint df_u\,df_v\; |f_u|\,\hat P_\phi(f_u,f_v)\, e^{i2\pi(f_u u+f_v v)} , \qquad (13)$$

which is exactly “2-D inverse FFT of ($|f_u|$ times the 2-D FFT of
$P_\phi$)”. Then

$$f(x,y,z) = \int_0^\pi g_\phi\big(x\cos\phi+y\sin\phi,\; z\big)\;d\phi . \qquad (14)$$

Equations (12)&ndash;(14) are the algorithm the code implements &mdash; with the
wrong multiplier in place of $|f_u|$.


<a id="why-the-multiplier-is-flat-along-f_v-slice-decoupling"></a>

# Why the multiplier is flat along $f_v$: slice decoupling

The fact that $W(f_u,f_v)=|f_u|$ does not depend on $f_v$ has a
clean consequence. Filtering along $f_u$ and the Fourier transform
along the *other* detector axis commute:

$$ \begin{aligned} g_\phi &= \mathcal{F}^{-1}_{u}\mathcal{F}^{-1}_{v}\Big[\,W(f_u)\,\hat P_\phi(f_u,f_v)\Big] \\ &= \mathcal{F}^{-1}_{u}\mathcal{F}^{-1}_{v}\Big[\,W(f_u)\,\mathcal{F}_{u}\mathcal{F}_{v}P_\phi\Big] = \mathcal{F}^{-1}_{u}\Big[\,W(f_u)\,\mathcal{F}_{u}P_\phi\Big] . \end{aligned} \qquad (15) $$

So the 2-D filtering decouples into **independent 1-D filtering of each
detector row** $P_\phi(\cdot,v)$ with the same ramp $|f_u|$.
Geometrically this is obvious from (4): every row $v=z$ is a
*complete* 2-D Radon data set for the slice $z=\mathrm{const}$, and
motion along $z$ is never projected onto the detector &mdash; it merely
*labels* which slice you are looking at. There is nothing to filter in
the $f_v$ direction; the correct weighting there is identically one.

It follows that 3-D FBP in this geometry is **exactly** 2-D FBP applied
slice by slice:

$$\text{3-D FBP with } |f_u| \;\equiv\; \text{2-D FBP of each sinogram } P_\phi(\cdot, z),\ \text{for every } z. \qquad (16)$$

This equivalence is not approximate; it is an identity, and it is the
sharpest available test of the implementation (see §7, Appendix B).


<a id="the-bug-an-isotropic-2-d-ramp-and-why-it-produces-saturation"></a>

# The bug: an isotropic 2-D ramp, and why it produces saturation

Until recently the package computed (`pyinverse/fbp3.py`)

    def ramp_filter3(grid_uv_ft_Hz):
        Cu_Hz, Cv_Hz = grid_uv_ft_Hz.centers
        return np.sqrt(Cu_Hz**2 + Cv_Hz**2)      # isotropic 2-D ramp — WRONG

Intuitively this is “the 2-D filter for a 2-D projection”, which is
exactly the natural-but-incorrect guess. Substituting (10),
$f_u^2 = f_x^2+f_y^2$ and $f_v = f_z$, so relative to the correct
result the reconstruction is reweighted by

$$\frac{W_{\text{code}}}{W_{\text{correct}}} = \frac{\sqrt{f_u^2+f_v^2}}{|f_u|} = \frac{\sqrt{f_x^2+f_y^2+f_z^2}}{\sqrt{f_x^2+f_y^2}} \;\geq\; 1 , \qquad (17)$$

which is unbounded in a neighbourhood of the $f_z$ axis ($f_u\to 0$)
and equals one only on the plane $f_v=f_z=0$.

Three consequences, each worth noting:

1.  **It is a Jacobian of the wrong integral.** $\sqrt{f_u^2+f_v^2}$ is
    the Jacobian of *polar coordinates in the 2-D detector-frequency
    plane* &mdash; i.e. of the 2-D reconstruction of a single projection,
    which is not what we are inverting.

2.  **It ramps *across* rows.** Unlike $|f_u|$, it depends on $f_v$, so
    it destroys the slice decoupling (15): it mixes the independent
    per-slice data sets and applies a $v$-varying weight. This is
    precisely the quantity that should have been left at one.

3.  **The bias is independent of the number of projections.** The gain (17)
    is a property of the *filter*, not of the data. Adding projections
    adds more equations, but each is reweighted by the same wrong gain,
    so the error does not converge away. This is the *saturation*:
    
    $$\text{NRMSE} \xrightarrow[\;N_\phi\to\infty\;]{} \text{const} \neq 0 . \qquad (18)$$
    
    The object-dependent part is the projection density; the filter part
    is not.

The fix is a single line (**applied**):

    def ramp_filter3(grid_uv_ft_Hz):
        Cu_Hz = grid_uv_ft_Hz.centers[0]
        return np.abs(Cu_Hz)                     # correct: 1-D ramp along f_u


<a id="why-this-survived-for-so-long"></a>

## Why this survived for so long

For a $z$-independent object (a cylinder, or any “slice-like”
phantom), $\hat P_\phi(f_u,f_v)$ is supported on the line $f_v=0$,
where $\sqrt{f_u^2+f_v^2} = |f_u|$ **exactly**. The two filters agree
there, so cylindrical test phantoms reconstruct identically under both.
Only a genuinely 3-D phantom exposes the error; the validation in §7
deliberately uses a stack of rotated ellipsoids for this reason.


<a id="numerical-evidence"></a>

# Numerical evidence

All figures below are produced by `tests/fbp_validation.py` on the
analytic ellipsoid/ellipse phantoms in this package (see §10 for how to
reproduce). Error is the relative $L_2$ norm against the analytic
object.

**2-D FBP (`pyinverse.fbp`) is correct** &mdash; monotone improvement with
projection count:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">\(N_\varphi\)</th>
<th scope="col" class="org-right">16</th>
<th scope="col" class="org-right">32</th>
<th scope="col" class="org-right">64</th>
<th scope="col" class="org-right">128</th>
<th scope="col" class="org-right">256</th>
<th scope="col" class="org-right">512</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">NRMSE</td>
<td class="org-right">0.83444</td>
<td class="org-right">0.47432</td>
<td class="org-right">0.21750</td>
<td class="org-right">0.12713</td>
<td class="org-right">0.12175</td>
<td class="org-right">0.12139</td>
</tr>
</tbody>
</table>

**3-D FBP with the original $\sqrt{f_u^2+f_v^2}$ filter saturates** &mdash;
flat in $N_\phi$, the direct fingerprint of the wrong Jacobian:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">\(N_\phi\)</th>
<th scope="col" class="org-right">8</th>
<th scope="col" class="org-right">16</th>
<th scope="col" class="org-right">32</th>
<th scope="col" class="org-right">64</th>
<th scope="col" class="org-right">128</th>
<th scope="col" class="org-right">256</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">NRMSE (\(\sqrt{f_u^2+f_v^2}\))</td>
<td class="org-right">2.83372</td>
<td class="org-right">2.79981</td>
<td class="org-right">2.79578</td>
<td class="org-right">2.79511</td>
<td class="org-right">2.79492</td>
<td class="org-right">2.79490</td>
</tr>

<tr>
<td class="org-left">NRMSE (\(\lvert f_u\rvert\))</td>
<td class="org-right">0.39129</td>
<td class="org-right">0.24269</td>
<td class="org-right">0.19509</td>
<td class="org-right">0.19223</td>
<td class="org-right">0.19214</td>
<td class="org-right">0.19214</td>
</tr>
</tbody>
</table>

With the correct $|f_u|$ the error falls monotonically and then floors
at the discretisation/resolution level (a resolution study at fixed
$N_\phi$ confirms the plateau $\approx 0.19$ is not a
projection-density effect). The wrong filter also inflates the
reconstruction amplitude ($\max|\text{recon}|\approx 3.85$ versus
$\approx 1.49$ for $|f_u|$, against a truth of $1.0$), consistent
with the gain (17) being $\geq 1$.

**Decisive equivalence check.** By (16), 3-D FBP with $|f_u|$ must equal
an independently written slice-by-slice 2-D FBP. It does:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">phantom</th>
<th scope="col" class="org-left">filter</th>
<th scope="col" class="org-right">NRMSE vs truth</th>
<th scope="col" class="org-left">relative \(L_2\) vs slice-by-slice 2-D FBP</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">ellipsoids (3-D)</td>
<td class="org-left">\(\sqrt{f_u^2+f_v^2}\)</td>
<td class="org-right">3.02419</td>
<td class="org-left">\(3.35\times10^{+0}\)</td>
</tr>

<tr>
<td class="org-left">ellipsoids (3-D)</td>
<td class="org-left">\(\lvert f_u\rvert\)</td>
<td class="org-right">0.20712</td>
<td class="org-left">\(1.7\times10^{-14}\)</td>
</tr>

<tr>
<td class="org-left">cylinder (z-invariant)</td>
<td class="org-left">\(\sqrt{f_u^2+f_v^2}\)</td>
<td class="org-right">0.18361</td>
<td class="org-left">\(6.6\times10^{-5}\)</td>
</tr>

<tr>
<td class="org-left">cylinder (z-invariant)</td>
<td class="org-left">\(\lvert f_u\rvert\)</td>
<td class="org-right">0.18364</td>
<td class="org-left">\(3.8\times10^{-16}\)</td>
</tr>
</tbody>
</table>

Note the last two rows: for the $z$-independent phantom both filters
agree to $\sim10^{-4}$, exactly as predicted in §6.

**Tilted detector.** The numerical evidence for $\theta_0 \neq 0$ is
collected in §8.


<a id="the-general-case-tilted-detector-theta_0-neq-0"></a>

# The general case: tilted detector, $\theta_0 \neq 0$

The code also admits a detector tilt (`theta0` in `fbp3_theta0`, with
basis vectors $\hat e_1=(\cos\phi,\sin\phi,0)$ and
$\hat e_2=(\sin\phi\sin\theta,-\cos\phi\sin\theta,\cos\theta)$).
Repeating the determinant computation of Appendix A with general
$\theta$ gives

$$\det\frac{\partial(f_x,f_y,f_z)}{\partial(\phi,f_u,f_v)} = -f_u\cos\theta , \qquad\text{so}\qquad d^3\xi = |f_u|\,|\cos\theta|\;d\phi\,df_u\,df_v . \qquad (19)$$

So $|f_u|$ remains the correct multiplier **up to a constant scale
factor** $|\cos\theta|$ for any $\theta\neq 90^\circ$. That constant is
**not** harmless: it is part of the inversion formula, and omitting it
inflates the reconstruction by exactly $\sec\theta_0$, uniformly in
space. `fbp3_theta0` applies `np.cos(theta0.rad)` for this reason.
Because the error is a pure amplitude, no study that varies only the
number of projections can see it &mdash; it takes a phantom whose data are
***complete*** at every tilt, i.e. one that is invariant along $z$.

However, the *coverage* changes. The plane normals are now
$\hat n(\phi)=\hat e_1\times\hat e_2=(\sin\phi\cos\theta,-\cos\phi\cos\theta,-\sin\theta)$,
whose tips trace a circle rather than filling a disc, and a frequency
$\boldsymbol{\xi}$ is sampled iff
$\boldsymbol{\xi}\cdot\hat n(\phi)=0$ for some $\phi$, i.e.

$$\cos\theta\,(f_x\sin\phi - f_y\cos\phi) = \sin\theta\, f_z \;\;\Longleftrightarrow\;\; \rho \,\geq\, |f_z \tan\theta|, \qquad \rho = \sqrt{f_x^2+f_y^2} . \qquad (20)$$

For $\theta=0$ this is satisfied everywhere except the measure-zero
$f_z$ axis (complete data). For $\theta\neq0$ the double cone

$$\rho < |f_z|\,|\tan\theta| \quad\Big(\text{i.e. within a half-angle } |\theta| \text{ of the } f_z\text{ axis}\Big)$$

is **never sampled** &mdash; a *missing cone*. A tilted rotating detector is
therefore an intrinsically incomplete-data problem, in contrast to the
exact $\theta=0$ case.

This branch is covered by the validation harness. On the $z$-invariant
control the tilted reconstruction matches an independent 2-D FBP of the
cross-section to $\sim10^{-4}$ at every tilt, and the amplitude is
$\theta$-independent to five digits; dropping the $\cos\theta_0$
factor breaks exactly that (the scale then drifts as $\sec\theta_0$).
On a genuinely 3-D phantom the error plateaus with more projections and
the plateau grows with $\theta_0$, tracking the energy fraction of the
missing cone: at $\theta_0 = 15^\circ, 30^\circ, 45^\circ$ the cone holds
$25.9\%, 43.3\%, 49.5\%$ of the (mean-removed) object energy and the
relative $L^2$ error settles at $0.542, 0.720, 0.816$, close to
$\sqrt{\text{cone fraction}}$. The `radon_matrices` path of
`fbp3_theta0` carries no $\theta$ dependence at all and is therefore
restricted to $\theta_0 = 0$ (enforced).


<a id="relation-to-fdk-and-other-geometries"></a>

# Relation to FDK and other geometries

The appearance of a **row-wise 1-D ramp**, rather than an isotropic 2-D
filter, is generic to geometries in which a planar detector rotates
about an axis. In cone-beam FDK (Feldkamp&ndash;Davis&ndash;Kress) one likewise
filters each detector row with a 1-D ramp (preceded by a cosine weight
and followed by a $1/r^2$ weight). The parallel-beam non-tilted case
treated here is the special case in which the filtered-backprojection
inversion is *exact*, with no weighting beyond the ramp and the angular
increment $\Delta\phi$.

A useful mental model: the Jacobian is always “$|\xi|$ resolved along
the direction in which the projections do *not* add information”, and
the ramp therefore lives only in the directions that the rotation
actually sweeps out.


<a id="reproducing"></a>

# Reproducing

The validation harness is headless and self-contained:

    python3 tests/fbp_validation.py

It runs the 2-D and 3-D sweeps, both filters, both phantoms, the
slice-by-slice equivalence check, and the tilted-detector
($\theta_0 \neq 0$) checks, and writes figures to `tests/out/`.
The package must be importable and the `lasserre` C extension built in
place (`python3 setup.py build_ext --inplace`); `scipy` is required.

The single line changed in `pyinverse/fbp3.py` was:

     def ramp_filter3(grid_uv_ft_Hz):
         """
         """
    -    Cu_Hz, Cv_Hz = grid_uv_ft_Hz.centers
    -    return np.sqrt(Cu_Hz**2 + Cv_Hz**2)
    +    Cu_Hz = grid_uv_ft_Hz.centers[0]
    +    return np.abs(Cu_Hz)

and the $\cos\theta_0$ factor of (19) is applied in `fbp3_theta0`:

    -    return phi_axis.rad.T * X_backproject
    +    return phi_axis.rad.T * np.cos(theta0.rad) * X_backproject

The `radon_matrices` path of `fbp3_theta0` encodes the untilted
geometry only &ndash; it carries no $\theta$ dependence &ndash; and is now
guarded accordingly.

After the change the harness reports (ellipsoid phantom,
$N_\phi = 64$) a relative $L_2$ error of $1.7\times10^{-14}$
against slice-by-slice 2-D FBP, and a direct check confirms
`ramp_filter3` equals $|f_u|$ and not $\sqrt{f_u^2+f_v^2}$.

Note that `notebooks/FBP3 trivial geometry.ipynb` calls `ramp_filter3`
directly; it now picks up the corrected filter automatically and should
be re-run to refresh any stored reconstructions.


<a id="appendix-a-the-determinant"></a>

# Appendix A: the determinant

With $\boldsymbol{\xi}(\phi,f_u,f_v)=f_u\hat e_1(\phi)+f_v\hat e_2$
and $$ \hat e_1=\begin{pmatrix}\cos\phi\\ \sin\phi\\ 0\end{pmatrix},\qquad \hat e_2=\begin{pmatrix}\sin\phi\sin\theta\\ -\cos\phi\sin\theta\\ \cos\theta\end{pmatrix}, $$ the three columns of the Jacobian are
$\partial_{\phi}\boldsymbol{\xi}$,
$\partial_{f_u}\boldsymbol{\xi}=\hat e_1$,
$\partial_{f_v}\boldsymbol{\xi}=\hat e_2$. Hence
$\det = \partial_\phi\boldsymbol{\xi}\cdot(\hat e_1\times\hat e_2)$
and

$$ \hat e_1\times\hat e_2 = \begin{pmatrix}\sin\phi\cos\theta\\ -\cos\phi\cos\theta\\ -\sin\theta\end{pmatrix}. $$

With $\partial_\phi\boldsymbol{\xi} = f_u(-\sin\phi,\cos\phi,0)^{\!\top} + f_v(\cos\phi\sin\theta,\sin\phi\sin\theta,0)^{\!\top}$ we get

$$ \begin{aligned} \det &= f_u\big(-\sin^2\phi\cos\theta-\cos^2\phi\cos\theta\big) + f_v\big(\cos\phi\sin\phi\sin\theta\cos\theta-\sin\phi\cos\phi\sin\theta\cos\theta\big)\\ &= -f_u\cos\theta . \end{aligned} $$

The $f_v$ terms cancel identically, which is the algebraic statement
of “the $f_v$ direction carries no Jacobian weight”. Setting
$\theta=0$ gives (11).


<a id="appendix-b-the-discrete-algorithm-as-implemented"></a>

# Appendix B: the discrete algorithm as implemented

For each rotation angle $\phi_i$ the code does, in effect:

    grid_uv_ft_i, p_uv_ft_i = grid_uv.spectrum(p_uv_i, real=True)   # 2-D FFT of the projection
    ramp                     = ramp_filter3(grid_uv_ft_i.Hz())       # the 2-D multiplier W(f_u,f_v)
    _, p_uv_filtered_i       = grid_uv_ft_i.ispectrum(p_uv_ft_i * ramp)  # 2-D inverse FFT
    X_backproject           += backproject3(theta0, phi_i, axes3, grid_uv, p_uv_filtered_i)

and finally multiplies by the angular increment, `X *` phi<sub>axis.rad.T</sub>=
(i.e. $\Delta\phi$, Eq. 14). The 2-D FFT, the 2-D multiplier, and the
backprojection are all as required; only the multiplier's profile is
wrong.

**Frequency convention.** `grid_uv_ft.Hz()` returns the grid with axes
scaled by $1/2\pi$, i.e. frequencies in cycles per unit length,
matching the $e^{+i2\pi\boldsymbol{\xi}\cdot\mathbf{x}}$ convention
used throughout this note and in `pyinverse`. The multiplier is
therefore $|f_u|$ in Hz, *not* $|\omega_u|$ in rad/s; the two differ
by $2\pi$. (Applying `Hz()` twice &mdash; e.g. inside a custom replacement
filter &mdash; silently introduces a factor $2\pi$, a trap worth
remembering when experimenting with filters.)


<a id="references"></a>

# References

1.  J. Radon, *Über die Bestimmung von Funktionen durch ihre
    Integralwerte längs gewisser Mannigfaltigkeiten*, Ber. Sächs. Akad.
    Wiss. 69 (1917), 262&ndash;277.
2.  F. Natterer, *The Mathematics of Computerized Tomography*, Wiley,
    1986 (3-D Fourier slice theorem and inversion; the Jacobian
    argument).
3.  A. C. Kak and M. Slaney, *Principles of Computerized Tomographic
    Imaging*, IEEE Press, 1988 (Ch. 3: 2-D FBP; Ch. 6: 3-D reconstruction
    from 2-D projections).
4.  L. A. Feldkamp, L. C. Davis, J. W. Kress, *Practical cone-beam
    algorithm*, J. Opt. Soc. Am. A 1 (1984), 612&ndash;619 (row-wise ramp
    filtering).
5.  P. Grangeat, *Mathematical framework of cone beam 3D reconstruction
    via the first derivative of the Radon transform*, in *Mathematical
    Methods in Tomography*, Springer LNM 1497 (1991) (the 3-D
    Fourier-plane relation).
6.  J. A. Fessler, *Image Reconstruction: Algorithms and Analysis* (in
    preparation), Ch. 2 and Ch. 3 &mdash; ellipsoid X-ray transforms and
    analytic projection formulas used to generate the phantom data here.

