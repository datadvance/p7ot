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

import openturns as ot
import da.p7core.gtapprox as gtapprox
import numpy as np


class ModelFunction(ot.NumericalMathFunction):
    def __new__(self, inputs_dimension, outputs_dimension, p7_model):
        if not isinstance(p7_model, gtapprox.Model):
            raise TypeError('no p7 model given. Expected ' + str(gtapprox.Model) + ' object')
        # Create an intermediate function to fill execution methods for NumericalMathFunction
        ot_python_function = ot.OpenTURNSPythonFunction(inputs_dimension, outputs_dimension)
        ot_python_function._exec = p7_model.calc
        ot_python_function._exec_sample = p7_model.calc
        function = ot.NumericalMathFunction(ot_python_function)
        gradient_implementation = _Gradient(inputs_dimension, outputs_dimension, p7_model)
        # Gradient object can't be passed directly because of misunderstanding with swig
        # Here the required gradient methods are implemented manually
        function.getGradient = lambda: gradient_implementation
        function.gradient = gradient_implementation.gradient
        function.getGradientCallsNumber = gradient_implementation.getCallsNumber
        function.setGradient(gradient_implementation)
        return function


class _Gradient(ot.NumericalMathGradientImplementation):
    def __init__(self, inputs_dimension, outputs_dimension, p7_model):
        self.__p7_model = p7_model
        self.__calls_number = 0
        # Shape of transposed Jacobian
        self.__inputs_dimension = inputs_dimension
        self.__outputs_dimension = outputs_dimension
        super(_Gradient, self).__init__()

    # Returns transposed Jacobian matrix for single point as openturns does
    def gradient(self, point):
        shape = np.array(point).shape
        # Check passed argument (should be 1D array)
        if len(shape) == 1 and shape[0] == self.__inputs_dimension:
            result = self.__p7_model.grad(point)
            self.__calls_number += 1
            return ot.Matrix(result).transpose()
        else:
            raise ValueError('Wrong shape %s of input array, required: (%s,)' % (shape, self.__inputs_dimension))

    def getCallsNumber(self):
        # Returns the gradient methods calls number
        return self.__calls_number

    def getInputDimension(self):
        # Returns NumericalMathFunction input dimension (the number of columns in Jacobian matrix)
        return self.__inputs_dimension

    def getOutputDimension(self):
        # Returns NumericalMathFunction output dimension (the number of rows in Jacobian matrix)
        return self.__outputs_dimension
