import numpy as np
from scipy.special import erf

def F_(x):
    return np.where(x==0, 1, erf(x)/x*np.pi**0.5/2)

class H2mol:
    def __init__(self, d, Z=1, alpha=None, coeff=None):
        """ Hydrogen molecule
        d = distance between protons
        Z = effective charge of a proton
        alpha = list of gaussian exponents
        coeff = list of gaussian coefficients
        """
        global h1,h2,J11,J22,J12,K

        self.d = d
        if alpha is None: alpha = [0.109818, 0.405771, 2.22766]
        if coeff is None: coeff = [0.444635, 0.535328, 0.154329]

        alpha = np.array(alpha) * Z**2
        c = np.array(coeff)
        L = len(alpha)
        beta = alpha.reshape(L,1) + alpha
        gamma = np.outer(alpha, alpha)/beta
        tau = d*np.sqrt(beta)
        rho = (4*gamma/beta)**0.75
        bb = beta * beta.reshape(L,L,1,1)
        sigma = d*np.sqrt(bb/(beta + beta.reshape(L,L,1,1)))
        lm = alpha*tau/beta
        mu = alpha.reshape(L,1,1)*sigma/beta.reshape(L,L,1,1)
        nu = sigma*np.abs(alpha.reshape(L,1,1,1) * alpha.reshape(L,1) -
                          alpha.reshape(L,1,1) * alpha)/bb

        pi2d = 2/np.pi**0.5/d
        cc = np.outer(c,c)
        rr = rho * rho.reshape(L,L,1,1)
        c4 = cc * cc.reshape(L,L,1,1)

        # overlap integral
        S = rho * np.exp(-gamma*d**2)
        # core Hamiltonian
        H11 = rho*(3*gamma - pi2d*tau*(1 + F_(tau)))
        H12 = S*(gamma*(3 - 2*gamma*d**2) - 2*pi2d*tau*F_(lm))
        # electron repulsion intgrals
        V1111 = rr * sigma
        V1122 = rr * sigma * F_(sigma)
        V1211 = rho * S.reshape(L,L,1,1) * sigma * F_(mu)
        V1212 = S   * S.reshape(L,L,1,1) * sigma * F_(nu)

        S = np.sum(cc * S)
        H11 = np.sum(cc * H11)
        H12 = np.sum(cc * H12)
        V1111 = pi2d * np.sum(c4 * V1111)
        V1122 = pi2d * np.sum(c4 * V1122)
        V1211 = pi2d * np.sum(c4 * V1211)
        V1212 = pi2d * np.sum(c4 * V1212)

        h1 = (H11 + H12)/(1+S)
        h2 = (H11 - H12)/(1-S)
        J11 = (V1111 + V1122 + 4*V1211 + 2*V1212)/(2*(1+S)**2)
        J22 = (V1111 + V1122 - 4*V1211 + 2*V1212)/(2*(1-S)**2)
        J12 = (V1111 + V1122 - 2*V1212)/(2*(1-S**2))
        K = (V1111 - V1122)/(2*(1-S**2))

    def energy(self, theta):
        c2 = np.cos(theta)**2
        s2 = np.sin(theta)**2
        E = (2*h1 + J11*c2)*c2 + (2*h2 + J22*s2)*s2
        E += 2*(J12 - 2*K)*s2*c2 + 1/self.d
        return E

    def eta(self):
        return (2*h2 - 2*h1 + J22 - J11)/(J11 + J22 - 2*J12 + 4*K)

    def theta_min(self):
        eta = self.eta()
        if eta >= 1: return 0
        else: return np.arccos(eta)/2

    def E_min(self):
        return self.energy(self.theta_min())


if __name__ == "__main__": H2mol.__init__(d = 0.75)
