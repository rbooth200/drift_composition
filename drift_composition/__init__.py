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
A 1D model for the advection and diffusion of molecules through protoplanetary
discs.
"""
__version__ = "0.1.0"

from drift_composition import adsorb_desorb
from drift_composition import advect_diffuse
from drift_composition import atoms
from drift_composition import constants
from drift_composition import disc
from drift_composition import grid
from drift_composition import molecule
