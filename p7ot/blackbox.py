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

from da.p7core import blackbox as p7blackbox
import numpy as np


class _Blackbox(p7blackbox.Blackbox):
    def __init__(self, blackbox, bounds):
        super(_Blackbox, self).__init__()
        self.__blackbox = blackbox
        # Variables
        self.__variables_dim = blackbox.getInputDimension()
        self.__variables_names = blackbox.getInputDescription()
        self.__variables_bounds = zip(bounds[0], bounds[1])
        # Responses
        self.__response_dim = blackbox.getOutputDimension()
        self.__response_names = blackbox.getOutputDescription()

    def prepare_blackbox(self):
        # Add variables
        for i in range(self.__variables_dim):
            self.add_variable(bounds=self.__variables_bounds[i], name=self.__variables_names[i])
        # Add responses
        for i in range(self.__response_dim):
            self.add_response(name=self.__response_names[i])

    def evaluate(self, points):
        result = []
        for point in points:
            result.append(list(self.__blackbox(point)))
        return result
