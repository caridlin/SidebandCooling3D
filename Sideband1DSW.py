#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals
from matplotlib import rc
import matplotlib.pyplot as plt
rc('font',**{'family':'serif','size'   : 12})
from qutip import *


# Basic constants
pi = 3.1415926
c = 299792458
h = 6.62606957e-34
hbar = h / 2 / pi
k_B = 1.3806488e-23
m_e = 9.10938291e-31
m_p = 1.672621898e-27
mu_0 = 4e-7 * pi
epsilon_0 = 1 / c**2 / mu_0
E = 1.602176565e-19
N_A = 6.02214129e+23


# Mass of Sr-88, Sr-87, Sr-86, Sr-84
M88 = 1.459706905272492E-25  
M87 = 1.4431557366419E-25  
M86 = 1.42655671117996E-25
M84 = 1.3934150821E-25
M = M88

# Fitting to get the cooling rate
from scipy.optimize import curve_fit

def f(x, a, b, c):
    return a * exp(-2 * pi * b * x) + c 

import timeit
from numpy import linspace, arange, divide, sqrt, exp
from scipy import stats

Nx = 15   # Dimension of x-harmonic oscillator
m = 10
omega_x = 2* pi * 200   # In units of kHz
delta = -2 * pi * 200   # In units of kHz
omega = 2 * pi * 50     # In units of kHz
gamma = 2 * pi * 7      # In units of kHz
eta = 0.15
g = basis(2,0)          # Ground state
e = basis(2,1)          # Excited state
x = fock(Nx, m)         # x-harmonic oscillator
phi = 0

# Recoil energy of the blue
Recoil_461 = (hbar * 2 * pi / (461E-9)) ** 2 / (hbar * 2 * M88 * 1000)
#print(Recoil_461)

# Destruction operators in 1d
a_x = destroy(Nx)

# H_eff / hbar, otherwise too small, too many floats
H = tensor(omega_x * (a_x.dag() * a_x + 0.5), qeye(2)) - tensor(qeye(Nx), 0.5 * delta * sigmaz()) + omega * 0.5 * (tensor((1j * eta * (a_x + a_x.dag())).expm() + (-1j * (eta * (a_x + a_x.dag()) + phi)).expm(), sigmap()) + tensor((-1j * eta * (a_x + a_x.dag())).expm() + (1j * (eta * (a_x + a_x.dag()) - phi)).expm(), sigmam()))

# Wavefunction in 1d, start from all in ground state
psi0 = tensor(x, g)

# Randomly generate u from dipole distribution
du = 0.01
u_ = arange(-1, 1, du)
p = [3 / 8 * (1 + u__) ** 2 for u__ in u_]
p = divide(p, sum(p))
dipole = stats.rv_discrete(name = 'custm', values = (range(len(p)), p))

dt = 100    # Time length
times = linspace(0.0, dt, 10000)
# N_iter = [1, 5, 10, 20]
# N_iter = [1, 5]
N_iter = [1]

import matplotlib.gridspec as gridspec
figure = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.3,hspace=0.4)
ax1 = plt.subplot(gs[0,0]) 
start = timeit.default_timer()

for N_iter_ in N_iter:
    idx = dipole.rvs(size = N_iter_)
    u = u_[idx]
    q = p[idx]
    q = divide(q, sum(q)) 
    q = sqrt(q)
    print(u, q)
    result = mesolve(H, psi0, times, [sqrt(gamma) * q[i] * tensor((1j * u[i] * eta * (a_x + a_x.dag())).expm(), sigmam()) for i in range(len(u))], [tensor(a_x.dag() * a_x, qeye(2))])
    ax1.plot(result.times, result.expect[0], label = '$N_{iter} = %d$' % (N_iter_), linewidth = 2, linestyle = '-');
    #popt, pcov = curve_fit(f, result.times[: 500], result.expect[0][: 500])
    #avg_fit = f(times, *popt)
    #print(pcov)
    #ax1.plot(times, avg_fit, label = r'Fit, $N_{iter} = %d$, initial cooling rate $\frac{dn}{dt} = %.2f kHz$' % (N_iter_, popt[0] * popt[1]), linewidth = 2, linestyle = '--');
    stop = timeit.default_timer()
    print(stop - start)     # Time spent in seconds
    
ax1.set_xlabel("$t (ms)$")
ax1.set_ylabel(r"$\langle n \rangle$")
#ax1.legend(loc = 1)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  ncol = 2)
ax1.set_title('Sideband cooling 1D SW\n'+
           '$\eta = %.2f, \omega = 2\pi %d$ kHz, $\omega_x = 2\pi %d$ kHz,  $\delta / \omega_x = %.1f, m=%d, N_x=%d$ \n' % (eta, omega / (2 * pi), omega_x / (2 * pi), delta / omega_x, m, Nx)
            + '$E_{r, 461} \sim 2\pi %.2f kHz$' % (Recoil_461 / (2 * pi)))
ax1.grid(1)
plt.savefig('Sideband_cooling_1D_SW_1.svg')