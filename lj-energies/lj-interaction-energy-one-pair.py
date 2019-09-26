
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

def lj(eps, sigma, r):
    return 4*eps * ((sigma / r) ** 12  - (sigma / r) ** 6)

lattice_a = 6.5

# CH4-Methane
eps_a = 158.500000
sigma_a = 2.720000

# Framework
eps_h = 513.264
# sigma_h = 1.052
# sigma_h = 1.663
# sigma_h = 2.274
# sigma_h = 2.884
sigma_h = 3.495

# use Lorentz-Bertholot mixing rules
eps = sqrt(eps_a * eps_h)
sigma = (sigma_a + sigma_h) / 2

dists = np.linspace(0,sqrt(3)*lattice_a,1000)[1:-1] #drop the first / last which will be inf

# # one H-A interaction
# energies = [lj(eps, sigma,r) for r in dists]

# lowest energy will be centered between two atoms H-A-H:
energies = [lj(eps, sigma,r) + lj(eps, sigma, sqrt(3)*lattice_a - r) for r in dists]

# print(energies[0], energies[250], energies[500], energies[750], energies[999])

min_energy = min(energies)
print('minimum_energy = %f' % min_energy)

fig = plt.figure(figsize=(8,8), tight_layout=True)
ax = fig.add_subplot(1, 1, 1)
# ax.set_yscale('log')
# ax.set_xlim(prop1range[0], prop1range[1])
ax.set_ylim(-1000,0)
ax.set_xlim(0,sqrt(3)*lattice_a)
ax.set_xticks(np.linspace(0,sqrt(3)*lattice_a,sqrt(3)*lattice_a + 1))
ax.set_xticks(np.linspace(0,sqrt(3)*lattice_a,sqrt(3)*lattice_a * 10 + 1), minor=True)
ax.grid(linestyle='-', color='0.8', zorder=0)
ax.plot(dists, energies, zorder=2)
fig.savefig("ljtest.png", dpi=288)
