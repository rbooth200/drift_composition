# drift_composition: A 1D model for the advection and diffusion of molecules
# through protoplanetary discs.
#
# Copyright (C) 2023  R. Booth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
"""
Models for the disc structure
"""
from collections import defaultdict
import numpy as np
from scipy.interpolate import PchipInterpolator

from drift_composition.grid import Grid
from drift_composition.advect_diffuse import FV_Model
from drift_composition.adsorb_desorb import (
    AdsorptionRate,
    ThermalDesorbtionRate,
    GrainModel,
)
from drift_composition.atoms import atoms_in_molecule, molecule_mass

from drift_composition.constants import k_boltzmann, m_hydrogen, G_Msun, Rau, Msun, yr


class DiscModel:
    """Model for a steady accretion disc.

    Parameters
    ----------
    grid : Grid object
        Radial grid for the disc
    Mdot : float, units= M_sun / year
        Gas mass accretion rate.
    alpha : function, alpha(R)
        Turbulent alpha parameters as function of R
    T : function, T(R)
        Temperature in K as a function of R
    Mstar : float, default=1, unit=Msun
        Stellar mass
    mu : float, default=1
        Mean molecular weight.
    """

    def __init__(self, grid, Mdot, alpha, T, Mstar=1, mu=2.35):
        self._grid = grid

        self._Mstar = Mstar
        self._Mdot = Mdot

        self._mu = mu
        self._T = T
        self._alpha = alpha

        self._compute_gas_surface_density()
        self.reset_molecules()

        self._FV = FV_Model(grid, stencil=self._stencil)
        self._FV_mol = FV_Model(grid, stencil=self._stencil, fields=2)

    _stencil = 3

    def aspect_ratio(self, R):
        """Disc aspect ratio H/R"""
        cs2 = k_boltzmann * self._T(R) / (self._mu * m_hydrogen)
        return np.sqrt(cs2) / self.v_k(R)

    def nu(self, R):
        """Kinematic viscosity."""
        cs2 = k_boltzmann * self._T(R) / (self._mu * m_hydrogen)
        return self._alpha(R) * cs2 / self.Omega_k(R)

    def Omega_k(self, R):
        """Keplerian angular frequency (in units of 2pi/year)"""
        return np.sqrt(G_Msun * self._Mstar / (R * R * R))

    def v_k(self, R):
        """Keplerian velocity"""
        return np.sqrt(G_Msun * self._Mstar / R)

    def v_gas(self, R):
        """Steady-state viscous velocity"""
        return -1.5 * self.nu(R) / R

    def v_dust(self, R, St):
        """Compute the radial drift velocity for a given Stokes number"""
        eta = self._lnP_interp(R, nu=1) * R * self.aspect_ratio(R) ** 2

        return (self.v_gas(R) + St * eta * self.v_k(R)) / (1 + St * St)

    def _compute_gas_surface_density(self):
        # Use the analytic solution
        g = self._grid
        self._Sigma_gas = (
            -(Msun / yr) * self._Mdot / (2 * np.pi * g.Rc * self.v_gas(g.Rc))
        )

        # Also store some other useful quantities:
        self._rho_mid = self._Sigma_gas / (
            np.sqrt(2 * np.pi) * self.aspect_ratio(g.Rc) * g.Rc
        )
        self._P_mid = (
            self._rho_mid * k_boltzmann * self._T(g.Rc) / (self._mu * m_hydrogen)
        )
        self._lnP_interp = PchipInterpolator(g.Rc, np.log(self._P_mid))

    def compute_dust_surface_density(self, Mdot_dust, St, Sc=1):
        """Compute the dust surface density and store it in self.Sigma_dust

        Parameters
        ----------
        Mdot_dust : float, units=Msun/year
            Accretion rate of dust
        St : callable,
            Stokes number as a function of radius.
        Sc : float (default=1) or array, shape=grid.size+1
            Schmidt Number determining dust diffusion coefficient D = nu/Sc
        Notes
        -----
         - Neglects feedback
        """
        self._Sigma_dust = np.zeros_like(self._grid.Rc, dtype="f8") # _update_dust needs this to exist
        self._update_dust(np.inf, Mdot_dust, St, Sc=Sc)

    def _update_dust(self, dt, Mdot_dust, St, Sc=1):
        """Update the dust surface density structure using an implicit step.

        Notes
        -----
        Setting dt=np.inf recovers the steady-state solution.
        """
        if dt == 0:
            return

        g = self._grid
        FV = self._FV

        adv = FV.create_advection_matrix(self.v_dust(g.Re, St(g.Re)))
        diff = FV.create_diffusion_matrix(self.nu(g.Re) / Sc, S=self._Sigma_gas)
        inv_dt = FV.create_linear_source_matrix(np.full_like(g.Rc, 1 / dt))
        mat = inv_dt - (adv + diff)

        source = g.cell_vol * self._Sigma_dust / dt
        source[-1] += Mdot_dust * Msun / yr

        self._St = St
        self._Sc = Sc
        self._Mdot_dust = Mdot_dust
        self._Sigma_dust = mat.solve(source)

    def compute_chemistry(self, molecules, abundances, d2g=0.01, grain_size=None):
        """Solve the transport of molecular species to compute their surface
        density structure.

        Parameters
        ----------
        molecules : list of Molecule
            Molecules to solve the structure for
        abundances : list of float
            Number abundance of the Molecules, relative to hydrogen.
        d2g : float, default=0.01
            Dust-to-gas mass ratio.
        Notes
        -----
        The accretion rate of the molecules is computed by working out
        what fraction of the molecule is frozen out and scaling the accretion
        rates of gas/dust by the amount of the species in each phase. When
        scaling by the dust accretion rate, the dust-to-gas ratio is used to
        determine whether the total flux of dust (and therefore ices) has been
        modified compared to the disc.
        """
        for mol in molecules:
            self._Sigma_mol[mol.name] = np.zeros([self._grid.size, 2], dtype="f8")
        self._update_molecules(np.inf, molecules, abundances, d2g, grain_size=grain_size)

    def _update_molecules(self, dt, molecules, abundances, d2g=0.01, grain_size=None):
        """Update molecular surface densities using an implicit step.

        Notes
        -----
        Setting dt=np.inf recovers the steady-state solution.
        """
        if dt == 0:
            return

        g = self._grid
        FV = self._FV_mol

        # Create transport matrices
        v_gas = self.v_gas(g.Re)
        v_dust = self.v_dust(g.Re, self.Stokes(g.Re))
        D = self.nu(g.Re) / self.Schmidt

        transport = (
            FV.create_advection_matrix(v_gas, 0)
            + FV.create_diffusion_matrix(D, S=self.Sigma_gas, field=0)
            + FV.create_advection_matrix(v_dust, 1)
            + FV.create_diffusion_matrix(D, S=self.Sigma_gas, field=1)
        )
        inv_dt = (
            FV.create_linear_source_matrix(np.full_like(g.Rc, 1 / dt), field=0)
            + FV.create_linear_source_matrix(np.full_like(g.Rc, 1 / dt), field=1)
        )

        # Set the grain properties
        grains = GrainModel()
        if grain_size is None:
            size = (2 / np.pi) * self.Stokes(g.Rc) * self.Sigma_gas / grains.rho
        else:
            size = np.full_like(g.Rc, grain_size)
        T, H = self._T(g.Rc), self.aspect_ratio(g.Rc) * g.Rc
        Sig_d = self.Sigma_dust

        for mol, abund in zip(molecules, abundances):
            # Create the rate objects
            ads = AdsorptionRate(mol, grains)
            des = ThermalDesorbtionRate(mol, grains)

            # Compute the mass fraction (assume total mass is H + He)
            mHe = (2 * self._mu - 4) / (4 - self._mu)
            mass_frac = abund * mol.mass_amu / (1 + mHe)
            # Estimate the fraction of each molecule on the grain surface
            # and use to specify the accretion rates
            if mol.p_stick > 0:
                ratio = (
                    des(Sig_d[-1] * mass_frac, Sig_d[-1], T[-1], H[-1], size[-1])[0]
                    / ads(Sig_d[-1] * 0, Sig_d[-1], T[-1], H[-1], size[-1])[0]
                )
                f_gas = ratio / (1 + ratio)
                f_dust = 1 / (1 + ratio)
            else:
                f_gas = 0
                f_dust = 1

            Mdot_vapor = self.Mdot_gas * f_gas * mass_frac
            Mdot_ice = self.Mdot_dust * f_dust * mass_frac / d2g

            # Since the desorption rate is non-linear we need to solve for
            # the surface density via iteration. The system may be written as:
            #   f(S) = (transport + ads(S) + des(S)) * S + Mdot
            # Taylor expand:
            #   f(S+dS) = (transport + ads(S) + des(S)) * (S+dS) + \
            #        (ads'(S) + des'(S)) * dS + Mdot
            #           =  [transport + ads(S) + des(S) + ads'(S) + des'(S)]*(S+dS)
            #        Mdot - (ads'(S) + des'(S)) * S
            # where f' denotes df(X)/dlog.
            #
            # We solve this via the Newton iteration:
            #   (S+dS) = [transport + ads(S) + des(S) + ads'(S) + des'(S)]^(-1) *
            #        ([ads'(S) + des'(S)] * S - Mdot)
            # with S=0 as the initial guess.

            Sigma = self._Sigma_mol[mol.name].copy()
            Mdot = g.cell_vol[:,None] * Sigma / dt

            Mdot[-1,0] += Mdot_vapor * Msun / yr
            Mdot[-1,1] += Mdot_ice * Msun / yr
            Mdot = Mdot.flatten()

            for i in range(100):
                # Create matrices for adsorption, desorption and jacobian
                R_ads, j_ads = ads(Sigma[:,0], Sig_d, T, H, size)
                R_des, j_des = des(Sigma[:,1], Sig_d, T, H, size)

                M = (
                    transport
                    + FV.create_mass_exchange_matrix(R_ads, 0, 1)
                    + FV.create_mass_exchange_matrix(R_des, 1, 0)
                )

                J_ads = FV.create_mass_exchange_matrix(j_ads, 0, 1)
                J_des = FV.create_mass_exchange_matrix(j_des, 1, 0)

                J = J_ads + J_des

                Sigma_new = (inv_dt - (M + J)).solve(Mdot - J.dot(Sigma.flatten()))
                Sigma_new = np.maximum(Sigma_new.reshape(-1, 2),0)


                diff = np.abs(Sigma_new - Sigma)
                abs_norm = np.maximum(Sigma_new.sum(1), Sigma.sum(1))[:, None] + 1e-300
                rel_norm = np.maximum(Sigma_new, Sigma) + 1e-300

                abs_err = diff / (1e-4 * abs_norm)
                rel_err = diff / (1e-4 * rel_norm + 1e-15*abs_norm)

                err = abs_err + rel_err

                Sigma = Sigma_new
                if err.max() < 1:
                    break
            else:
                raise RuntimeError(f"Run did not converge for {mol.name}")

            self._Sigma_mol[mol.name] = Sigma
        self._molecules = self._molecules.union(set(molecules))

    def compute_elemental_column(self, dust=None):
        """Compute the column density of each element.

        Parameters
        ----------
        dust : dict, default=None
            Composition of the dust grains, in the form
                { molecule : abundance}.
            Only the relative abundances are needed as everything is scaled to
            the total mass.  If None, the solid composition is computed using
            the ice composition only.

        Returns
        -------
        column : dict, units=cm^-2
            The column density of each element included.

        Notes
        -----
        The hydrogen column is always computed directly from the gas surface
        density and the mean molecular weight.
        """
        # Compute column density of each num
        column = defaultdict(lambda: np.zeros([self._grid.size, 2], dtype="f8"))
        for mol in self._molecules:
            elements = mol.get_atoms()
            for el in elements:
                column[el] += elements[el] * self._Sigma_mol[mol.name] / mol.mass

        # Compute the H column
        H = np.zeros([self._grid.size, 2], dtype="f8")

        xHe = (0.5 * self._mu - 1) / (4 - self._mu)
        H[:, 0] = self.Sigma_gas / (self._mu * m_hydrogen * (0.5 + xHe))

        column["H"] = H

        # Compute the dust column:
        if dust:
            m_dust = 0
            for mol in dust:
                m_dust += molecule_mass(mol) * m_hydrogen * dust[mol]
            for mol in dust:
                elements = atoms_in_molecule(mol)
                for el in elements:
                    column[el][:, 1] += (
                        elements[el] * dust[mol] * self.Sigma_dust / m_dust
                    )

        return column

    def compute_molecular_column(self):
        """Compute the column density of each molecule.

        Returns
        -------
        column : dict, units=cm^-2
            The column density of each molecule included.

        Notes
        -----
        The hydrogen column is always computed directly from the gas surface
        density and the mean molecular weight.
        """
        # Compute column density of each num
        column = defaultdict(lambda: np.zeros([self._grid.size, 2], dtype="f8"))
        for mol in self._molecules:
            column[mol.name] = self._Sigma_mol[mol.name] / mol.mass

        # Compute the H2 column
        H = np.zeros([self._grid.size, 2], dtype="f8")

        xHe = (0.5 * self._mu - 1) / (4 - self._mu)
        H[:, 0] = self.Sigma_gas / (self._mu * m_hydrogen * (0.5 + xHe))

        column["H2"] = H / 2

        return column

    def reset_molecules(self):
        """Remove the results for any molecules currently calculated"""
        self._molecules = set()
        self._Sigma_mol = {}

    @property
    def Sigma_gas(self):
        return self._Sigma_gas

    @property
    def Sigma_dust(self):
        return self._Sigma_dust

    @property
    def Sigma_mol(self):
        return self._Sigma_mol

    @property
    def Molecules(self):
        return self._molecules

    @property
    def Stokes(self):
        return self._St

    @property
    def Schmidt(self):
        return self._Sc

    @property
    def grid(self):
        return self._grid

    @property
    def Mdot_gas(self):
        return self._Mdot

    @property
    def Mdot_dust(self):
        return self._Mdot_dust


class TimeDependentDiscModel(DiscModel):
    """Model for time-dependent chemiscal transport in a disc.

    Parameters
    ----------
    grid : Grid object
        Radial grid for the disc
    Mdot : float, units= M_sun / year
        Gas mass accretion rate.
    alpha : function, alpha(R)
        Turbulent alpha parameters as function of R
    T : function, T(R)
        Temperature in K as a function of R
    Mstar : float, default=1, unit=Msun
        Stellar mass
    mu : float, default=1
        Mean molecular weight.
    """

    def __init__(self, grid, Mdot, alpha, T, Mstar=1, mu=2.35):
        super().__init__(grid, Mdot, alpha, T, Mstar=Mstar, mu=mu)

        self._age = 0


    def init(self, dust_to_gas, St, molecules, abundances, Sc=1):
        """Initialize the dust and molecules, assuming steady-state accretion."""
        self.compute_dust_surface_density(dust_to_gas * self.Mdot_gas, St, Sc=Sc)
        self.compute_chemistry(molecules, abundances, d2g=dust_to_gas)

        self._dust_to_gas = dust_to_gas
        self._abundances = abundances

    def update(self, dt, Mdot_dust, St):
        """Update the disc structure by a time-step dt.

        Parameters
        ----------
        dt : float, units=yr/2pi
            Time-step to evolve the disc.
        Mdot_dust : float, units=Msun/year
            Accretion rate of dust
        St : callable,
            Stokes number as a function of radius.
        """
        self._age += dt

        self._update_dust(dt, Mdot_dust, St, Sc=self.Schmidt)
        self._update_molecules(dt, self.Molecules, self._abundances, d2g=self._dust_to_gas)

    @property
    def age(self):
        return self._age