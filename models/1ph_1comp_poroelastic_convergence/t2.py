import numpy as np
from sympy import *
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
#from sympy.vector import divergence, CoordSys3D

class RhsPoroelastic:
    def __init__(self, stf, biot, perm, visc, grav, rho_f):
        self.stf = Matrix(np.array(stf).reshape(6,6))
        self.biot = Matrix(np.array(biot).reshape(3,3))
        self.perm = Matrix(np.array(perm).reshape(3,3))
        self.grav = grav
        self.rho_f = rho_f

        self.x, self.y, self.z, self.t = symbols('x y z t')
        u = ((self.x - 0.5) ** 2 - self.y - self.z) * (1 + self.t ** 2)
        v = ((self.y - 0.5) ** 2 - self.x - self.z) * (1 + self.t ** 2)
        w = ((self.z - 0.5) ** 2 - self.x - self.y) * (1 + self.t ** 2)
        p = 1 / 2 / sin(1) * sin((1-self.x) * (1-self.y) * (1-self.z)) + (1-self.x)**3 * (1-self.y)**2 * (1-self.z) * (1+self.t**2) / 2
        # Mechanics
        disp = Matrix([u, v, w])
        dudx = disp.jacobian([self.x, self.y, self.z])
        eps = Matrix([dudx[0,0], dudx[1,1], dudx[2,2], dudx[1,2] + dudx[2,1], dudx[0,2] + dudx[2,0], dudx[0,1] + dudx[1,0]])
        sig = self.stf * eps
        a = p * self.biot
        f_biot = Matrix([diff(a[0, 0], self.x) + diff(a[0, 1], self.y) + diff(a[0, 2], self.z),
                         diff(a[1, 0], self.x) + diff(a[1, 1], self.y) + diff(a[1, 2], self.z),
                         diff(a[2, 0], self.x) + diff(a[2, 1], self.y) + diff(a[2, 2], self.z)])
        self.f = -Matrix([diff(sig[0], self.x) + diff(sig[5], self.y) + diff(sig[4], self.z),
                    diff(sig[5], self.x) + diff(sig[1], self.y) + diff(sig[3], self.z),
                    diff(sig[4], self.x) + diff(sig[3], self.y) + diff(sig[2], self.z)])
        self.f += f_biot
        # stress tensor
        self.total_stress = sig
        self.total_stress[0] -= a[0, 0]
        self.total_stress[1] -= a[1, 1]
        self.total_stress[2] -= a[2, 2]
        self.total_stress[3] -= a[1, 2]
        self.total_stress[4] -= a[0, 2]
        self.total_stress[5] -= a[0, 1]
        # Flow
        self.acc = diff(p, self.t)
        self.darcy_flow = -TC.darcy_constant / visc * self.perm * Matrix([diff(p, self.x), diff(p, self.y), diff(p, self.z)])
        self.darcy_flow_grav = TC.darcy_constant / visc * self.perm * self.rho_f * self.grav * Matrix([0, 0, 1])
        self.darcy_velocity = visc * (self.darcy_flow + self.darcy_flow_grav)
        phi = self.darcy_flow + self.darcy_flow_grav + self.biot * Matrix([diff(u, self.t), diff(v, self.t), diff(w, self.t)])
        self.flow = diff(phi[0], self.x) + diff(phi[1], self.y) + diff(phi[2], self.z)

        self.f_func = lambdify((self.x, self.y, self.z, self.t), self.f, 'numpy')
        self.acc_func = lambdify((self.x, self.y, self.z, self.t), self.acc, 'numpy')
        self.flow_func = lambdify((self.x, self.y, self.z, self.t), self.flow, 'numpy')
        self.total_stress_func = lambdify((self.x, self.y, self.z, self.t), self.total_stress, 'numpy')
        self.darcy_velocity_func = lambdify((self.x, self.y, self.z, self.t), self.darcy_velocity, 'numpy')

class RhsThermoporoelastic:
    def __init__(self, stf, biot, perm, th_expn, heat_cond,
                 visc, grav, rho_f, th_expn_poro, heat_capacity):
        self.stf = Matrix(np.array(stf).reshape(6,6))
        self.biot = Matrix(np.array(biot).reshape(3,3))
        self.perm = Matrix(np.array(perm).reshape(3,3))
        self.th_expn = Matrix(np.array(th_expn).reshape(3,3))
        self.heat_cond = Matrix(np.array(heat_cond).reshape(3,3))
        self.grav = grav
        self.rho_f = rho_f
        self.th_expn_poro = th_expn_poro
        self.heat_capacity = heat_capacity

        self.x, self.y, self.z, self.t = symbols('x y z t')
        u = ((self.x - 0.5) ** 2 - self.y - self.z) * (1 + self.t ** 2)
        v = ((self.y - 0.5) ** 2 - self.x - self.z) * (1 + self.t ** 2)
        w = ((self.z - 0.5) ** 2 - self.x - self.y) * (1 + self.t ** 2)
        p = 3 - self.x - self.y - self.z
        t = 1 / 2 / sin(1) * sin((1-self.x) * (1-self.y) * (1-self.z)) + (1-self.x)**3 * (1-self.y)**2 * (1-self.z) * (1+self.t**2) / 2

        # Mechanics
        disp = Matrix([u, v, w])
        dudx = disp.jacobian([self.x, self.y, self.z])
        eps = Matrix([dudx[0,0], dudx[1,1], dudx[2,2], dudx[1,2] + dudx[2,1], dudx[0,2] + dudx[2,0], dudx[0,1] + dudx[1,0]])
        sig = self.stf * eps
        a = p * self.biot
        f_biot = Matrix([diff(a[0, 0], self.x) + diff(a[0, 1], self.y) + diff(a[0, 2], self.z),
                         diff(a[1, 0], self.x) + diff(a[1, 1], self.y) + diff(a[1, 2], self.z),
                         diff(a[2, 0], self.x) + diff(a[2, 1], self.y) + diff(a[2, 2], self.z)])
        b = t * self.th_expn
        f_th = Matrix([diff(b[0, 0], self.x) + diff(b[0, 1], self.y) + diff(b[0, 2], self.z),
                         diff(b[1, 0], self.x) + diff(b[1, 1], self.y) + diff(b[1, 2], self.z),
                         diff(b[2, 0], self.x) + diff(b[2, 1], self.y) + diff(b[2, 2], self.z)])
        self.f = -Matrix([diff(sig[0], self.x) + diff(sig[5], self.y) + diff(sig[4], self.z),
                    diff(sig[5], self.x) + diff(sig[1], self.y) + diff(sig[3], self.z),
                    diff(sig[4], self.x) + diff(sig[3], self.y) + diff(sig[2], self.z)])
        self.f += f_biot + f_th
        # Flow
        self.acc = diff(p, self.t)
        phi = -TC.darcy_constant / visc * self.perm * (Matrix([diff(p, self.x), diff(p, self.y), diff(p, self.z)]) -
                    self.rho_f * self.grav * Matrix([0, 0, 1])) + \
                    self.biot * Matrix([diff(u, self.t), diff(v, self.t), diff(w, self.t)])
        self.flow = diff(phi[0], self.x) + diff(phi[1], self.y) + diff(phi[2], self.z)
        # Energy
        self.energy = diff(self.heat_capacity * t, self.t)

        self.f_func = lambdify((self.x, self.y, self.z, self.t), self.f, 'numpy')
        self.acc_func = lambdify((self.x, self.y, self.z, self.t), self.acc, 'numpy')
        self.flow_func = lambdify((self.x, self.y, self.z, self.t), self.flow, 'numpy')

