#
# coding: utf-8
#
# Copyright (C) DATADVANCE, 2016
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# along with this library.  If not, see <http://www.gnu.org/licenses/>.
#

"""
p7ot
====
    p7ot is a python package for pSeven Core integration in OpenTURNS

Using
-----
    Just write in Python
    >>> import openturns as ot
    >>> import p7ot
    >>> experiment = p7ot.LHS(bounds=ot.Interval([0]*3, [10]*3), count=10, useOptimized=True)
    >>> print experiment.generate()
"""


from .gtdoe import (
    Sequence,
    LHS,
    BoxBehnken,
    FullFactorial,
    FractionalFactorial,
    OptimalDesign,
    OrthogonalArray,
    ParametricStudy,
    AdaptiveBlackbox,
    AdaptiveSample,
    AdaptiveLHS
)
from .gtopt import GTOpt
from .gtapprox import ModelFunction


__version__ = 1.0
