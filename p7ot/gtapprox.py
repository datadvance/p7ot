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
    """
    Approximation model.

    Parameters
    ----------
    inputs_dimension: positive int
        The inputs dimension.
    outputs_dimension: positive int
        The outputs dimension.
    p7_model: :class:`~da.p7core.gtapprox.Model`
        p7core approximation model.

    Notes
    -----
    You have to build approximation model by means of p7core to use it as NumericalMathFunction.

    Examples
    --------
    >>> import numpy as np
    >>> import da.p7core.gtapprox as gtapprox
    >>> import p7ot
    >>> inputs = np.random.random((30, 2))
    >>> outputs = [[x1*x1 + x2*x2] for (x1, x2) in inputs]
    >>> p7_model = gtapprox.Builder().build(inputs, outputs)
    >>> function = p7ot.ModelFunction(p7_model)
    >>> print function([2, 2])
    [7.95365]
    """
    def __new__(self, p7_model):
        if not isinstance(p7_model, gtapprox.Model):
            raise TypeError('No p7 model given. Expected ' + str(gtapprox.Model) + ' object')
        # Create an intermediate function to fill execution methods for NumericalMathFunction
        ot_python_function = ot.OpenTURNSPythonFunction(p7_model.size_x, p7_model.size_f)
        ot_python_function._exec = p7_model.calc
        ot_python_function._exec_sample = p7_model.calc
        result = ot.NumericalMathFunction(ot_python_function)
        # Gradient object can't be passed directly
        # Here the required gradient methods are implemented manually
        gradient_implementation = _Gradient(p7_model.size_x, p7_model.size_f, p7_model)
        result.getGradient = lambda: gradient_implementation
        result.gradient = gradient_implementation.gradient
        result.getGradientCallsNumber = gradient_implementation.getCallsNumber
        result.setGradient(gradient_implementation)
        return result


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
        # Get the calls number
        return self.__calls_number

    def getInputDimension(self):
        # Get the number of the inputs (columns in Jacobian matrix).
        return self.__inputs_dimension

    def getOutputDimension(self):
        # Get number of the outputs (rows in Jacobian matrix)
        return self.__outputs_dimension
