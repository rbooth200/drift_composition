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
Basic routines for solving the advection-diffusion equations
"""
import numpy as np

from drift_composition import grid
from drift_composition.reconstruct import (
    compute_FV_weights,
    construct_FV_edge_weights,
    join_symmetric_stencil,
    sparse_dot_product,
    build_sparse_matrix,
    BandedMatrix,
)


class FV_Model:
    r"""
    Creates matrix representations for finite-volume equations


    Creates the matrix-structure for a 1D advection-diffusion model:
        dQ/dt + \div (Q v) = \div [D . S \grad (Q/S)]

    Parameters
    ----------
    grid : grid object
        Describes the grid structure of the disc
    stencil : int, default=2
        Number of neighbours used either side of the target cell in the
        reconstruction. For reference, stencil = 1 is second order, piece-wise
        linear reconstruction, stencil = 2 is piece-wise cubic.
    fields : int, default=1
        The number of different fields (variables) used in the model.

    Notes
    -----
    - Reconstruction is done without any slope limiting.
    - Reconstruction at the boundaries is done at lower order.
    """

    def __init__(self, grid, stencil=2, fields=1):
        self._grid = grid
        self._stencil = stencil
        self._fields = fields

        # Setup the weights for reconstructing Q and it's derivative to the
        # cell edges
        wp, wm = construct_FV_edge_weights(grid.Re, grid.order, stencil)
        self._weights = join_symmetric_stencil(wp, wm)

    @property
    def grid(self):
        return self._grid

    @property
    def stencil(self):
        return self._stencil

    def create_advection_matrix(self, v_edge, field=0):
        r"""Determine the matrix elements for a 1D-advection model:
            dQ/dt = - \div (Q v).

        Strictly, this function evaluates the volume-integrated right-
        hand-side of this using the flux-conservative expression:
        \int dQ/dt dV  = - \int Q v \cdot dA

        Parameters
        ----------
        v_edge : array, length=grid.size+1
            Velocity at the cell edges.
        field : int, default=0,
            Field to create the matrix for.

        Returns
        -------
        advection_matrix : BandedMatrix
            Matrix defining the coefficients in - \int Q v \cdot dA.

        Notes
        -----
        - Inflow boundaries are not treated, but can be included by adding
        a source term to the inner or outer-most cell equal to the desired flux
        of Q across the boundary.
        - It is be necessary to add some diffusion because a central scheme is
        used.
        """
        # Use the reconstruction for Q to compute the fluxes across each face
        fluxes = (v_edge * self.grid.face_area)[:, None] * self._weights[:, 0]
        if v_edge[0] >= 0:
            fluxes[0] = 0
        if v_edge[-1] <= 0:
            fluxes[-1] = 0

        # Take the difference:
        mat = np.zeros([fluxes.shape[0] - 1, fluxes.shape[1] + 1], dtype="f8")
        mat[:, :-1] = fluxes[:-1]
        mat[:, 1:] -= fluxes[1:]

        return self._create_banded(
            np.arange(-self.stencil, self.stencil + 1), mat, field
        )

    def create_diffusion_matrix(self, D, field=0, S=None, close_boundaries=3):
        r"""Determine the matrix elements for a 1D-diffusion model:
            dQ/dt =  \div [D S \grad (Q/S)].

        Strictly, this function evaluates the volume-integrated right-
        hand-side of this using the flux-conservative expression:
        \int dQ/dt dV  =  \int D S \grad (Q/S) \cdot dA

        Parameters
        ----------
        D : array, length=grid.size+1
            Diffusion coefficient at the cell edges.
        field : int, default=0,
            Field to create the matrix for.
        S : array, length=grid.size, optional
            Optional argument to allow the diffusion to model 'tracer'
            diffusion (Fick's Law). If neglected S=1 is used.
        close_boundaries : int, default=3:
            Determines which boundaries are closed:
                0 : Neither
                1 : Inner boundary only
                2 : Outer boundary only
                3 : Both

        Returns
        -------
        diffusion_matrix : BandedMatrix
            Matrix defining the coefficients in -\int D S \grad (Q/S) \cdot dA.


        Notes
        -----
        - Inflow boundaries are not treated, but can be included by adding
        a source term to the inner or outer-most cell equal to the desired flux
        of Q across the boundary.
        """
        if S is None:
            S = np.ones(self.grid.size, dtype="f8")

        # Weights for reconstruction of quantitiy and it's gradient.
        w0, w1 = self._weights[:, 0], self._weights[:, 1]

        # Break down the matrix into 2 terms:
        # flux = - D \nabla Q + D (\nabla S)/S Q
        s = np.arange(-self.stencil, self.stencil)
        grad_S = sparse_dot_product(w1, S, s) / sparse_dot_product(w0, S, s)

        fluxes = (D * self.grid.face_area)[:, None] * (-w1 + grad_S[:, None] * w0)

        if close_boundaries not in range(4):
            raise ValueError("close_boundaries must be in range(4)")
        else:
            if close_boundaries & 1:
                fluxes[0] = 0
            if close_boundaries & 2:
                fluxes[-1] = 0

        # Take the difference:
        mat = np.zeros([fluxes.shape[0] - 1, fluxes.shape[1] + 1], dtype="f8")
        mat[:, :-1] = fluxes[:-1]
        mat[:, 1:] -= fluxes[1:]

        return self._create_banded(
            np.arange(-self.stencil, self.stencil + 1), mat, field
        )

    def create_mass_exchange_matrix(self, rate, field_1, field_2):
        r"""Create a source term matrix of the form
            dQ_1/dt = -rate*Q1
            dQ_2/dt =  rate*Q1

        The integral form
            \int dQ_1/dt dV  = - \int rate * Q1 dV
            \int dQ_2/dt dV  = + \int rate * Q1 dV
        will be used.

        Parameters
        ----------
        rate : array, size=grid.size
            The rate coefficient of the linear source term
            evaluated at the cell centers.
        field_1 : int,
            The field losing mass
        field_2 : int,
            The field gaining mass
        """
        stencil = np.array([0, field_1 - field_2])
        weights = np.zeros([self.grid.size * self._fields, 2], dtype="f8")

        weights[field_1 :: self._fields, 0] = -rate * self.grid.cell_vol
        weights[field_2 :: self._fields, 1] = +rate * self.grid.cell_vol

        return BandedMatrix.create_from_weights(stencil, weights)

    def create_linear_source_matrix(self, rate, field=0):
        r"""Create a source term matrix of the form
            dQ/dt = rate * Q
        The integral form
            \int dQ/dt dV  = \int rate * Q dV
        will be used.

        Parameters
        ----------
        rate : array, size=grid.size
            The rate coefficient of the linear source term
            evaluated at the cell centers.
        field : int, default=0
            Field to create the matrix for.

        Returns
        -------
        source_matrix : BandedMatrix
            Matrix defining the coefficients in \int rate * Q dV.

        Notes
        -----
         - A first order reconstruction is used.
        """
        return self._create_banded((0,), (rate * self.grid.cell_vol)[:, None], field)

    def get_field_ids(self, field):
        """Get the indices that select the variables corresponding to the field."""
        return np.arange(self._grid.size) * self._fields + field

    def _create_banded(self, stencil, matrix, field):
        """Make the banded matrix from the interpolation coefficients."""
        if field >= self._fields or field < 0:
            raise ValueError(f"Can only make matrices field < {self._fields}")

        # Correct stencil for number of fields:
        stencil = np.array(stencil) * self._fields

        field_mat = np.zeros(
            [matrix.shape[0] * self._fields, matrix.shape[1]], matrix.dtype
        )

        field_mat[field :: self._fields] = matrix

        return BandedMatrix.create_from_weights(stencil, field_mat)
