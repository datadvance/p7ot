from da.p7core import gtapprox
import unittest
import numpy as np
import math
from ..gtapprox import ModelFunction
from ..gtopt import GTOpt
import openturns as ot


class TestGTApprox(unittest.TestCase):

    def sphere(self, inputs):
        output = 0
        for x in inputs:
            output += x*x
        return output

    def rastrigin(self, inputs):
        dimension = 10*len(inputs)
        output = dimension
        for x in inputs:
            output += x*x-10*math.cos(2*math.pi*x)
        return output

    def test_Model(self):
        input_dim = 3
        output_dim = 2
        count = 30
        inputs = np.random.random((count, input_dim))
        outputs = [[self.sphere(x), self.rastrigin(x)] for x in inputs]
        # p7core model
        p7_model = gtapprox.Builder().build(inputs, outputs)
        # p7ot model function
        model_function = ModelFunction(p7_model)
        # ot function
        ot_function = ot.NumericalMathFunction(model_function)
        self.assertEqual(ot_function.getInputDimension(), input_dim)
        self.assertEqual(ot_function.getOutputDimension(), output_dim)
        # Compare the result function and p7core model outputs
        for x in inputs:
            self.assertTrue(np.array_equal(p7_model.calc(x), ot_function(x)))
