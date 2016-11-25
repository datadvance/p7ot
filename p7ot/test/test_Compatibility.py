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

import unittest
import numpy as np
import math
import openturns as ot
from openturns.viewer import View
from da.p7core import gtapprox
from da.p7core import gtopt
from ..gtdoe import Sequence, AdaptiveBlackbox
from ..gtapprox import ModelFunction
from ..gtopt import GTOpt
from distutils.version import LooseVersion


class TestCompatibility(unittest.TestCase):

    def test_gtdoe(self):
        # Experiment implementation
        p7ot_experiment = Sequence(count=10, bounds=ot.Interval(3))
        p7ot_result = p7ot_experiment.generate()
        if LooseVersion(ot.__version__) >= LooseVersion('1.8') and LooseVersion(ot.__version__) != LooseVersion('1.8rc1'):
            myExperiment = ot.Experiment(p7ot_experiment)
            myExperimentImplementation = myExperiment.getImplementation()
            mySecondExperiment = ot.Experiment()
            mySecondExperiment.setImplementation(myExperimentImplementation)
            self.assertEqual(p7ot_result, mySecondExperiment.generate())
        else:
            with self.assertRaises(NotImplementedError):
                ot.Experiment(p7ot_experiment)
        # Using the NumericalMathFunction in the AdaptiveBlackbox DoE
        function = ot.NumericalMathFunction(['x1', 'x2'], ['x1^2 + x2^2'])
        experiment = AdaptiveBlackbox(count=10, bounds=ot.Interval(2), blackbox=function)
        experiment.generate()
        self.assertTrue(isinstance(experiment.getBlackbox(), ot.NumericalMathFunction))

    def test_gtapprox(self):
        input_dim = 2
        output_dim = 1
        count = 10
        inputs = np.random.random((count, input_dim))
        outputs = [[x1*x1 + x2*x2] for (x1, x2) in inputs]
        # p7core model
        p7_model = gtapprox.Builder().build(inputs, outputs)
        # p7ot model function
        p7ot_function = ModelFunction(p7_model)
        # openturns function
        ot_function = ot.NumericalMathFunction(p7ot_function)
        # openturns optimization problem
        ot_optimization_problem = ot.OptimizationProblem()
        ot_optimization_problem.setObjective(p7ot_function)
        ot_optimization_problem.setBounds(ot.Interval([-100]*2, [100]*2))
        p7ot_solver = GTOpt(ot_optimization_problem)
        p7ot_solver.run()
        # openturns graphic
        input_dim = 1
        output_dim = 1
        count = 10
        inputs = np.random.uniform(0, 10, [count, input_dim])
        outputs = [math.sin(x) for x in inputs]
        p7ot_function = ModelFunction(gtapprox.Builder().build(inputs, outputs))
        graph = p7ot_function.draw(0, 10, 100)
        # View(graph).show()

    def test_gtopt(self):
        # Schaffer function
        objective = ot.NumericalMathFunction(['x', 'y'], ['0.5 + ((sin(x^2-y^2))^2-0.5)/((1+0.001*(x^2+y^2))^2)'])
        lb = [-10]*objective.getInputDimension()
        ub = [10]*objective.getInputDimension()
        # Solve by p7ot.GTOpt
        problem = ot.OptimizationProblem()
        problem.setObjective(objective)
        problem.setBounds(ot.Interval(lb, ub))
        solver = GTOpt(problem)
        solver.setStartingPoint(ub)
        solver.run()
        ot_result = solver.getResult()
        # Solve by p7core.gtopt.Solver

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                for i in range(objective.getInputDimension()):
                    self.add_variable(bounds=(lb[i], ub[i]), initial_guess=ub[i])
                self.add_objective()

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    functions_batch.append(objective(x))
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        p7_problem = Problem()
        p7_result = gtopt.Solver().solve(problem=p7_problem)
        # Compare results
        self.assertEqual(ot_result.getOptimalValue(), p7_result.optimal.f[0])
        self.assertEqual(ot_result.getOptimalPoint(), p7_result.optimal.x[0])
        self.assertEqual(ot_result.getIterationNumber(), len(p7_problem.history))

if __name__ == '__main__':
    unittest.main()
