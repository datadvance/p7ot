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
import unittest
import numpy as np
from da.p7core import gtopt
from ..gtopt import GTOpt


class TestGTOpt(unittest.TestCase):

    def compare_results(self, p7ot_result, problem, maximization=False, level_value=None):
        # Solve optimization problem
        p7_result = gtopt.Solver().solve(problem)
        # Compare results
        keys = p7ot_result.optimal.__dict__.keys()
        p7ot_values = p7ot_result.optimal.__dict__.values()
        p7_values = p7_result.optimal.__dict__.values()
        for key, p7ot_value, p7_value in zip(keys, p7ot_values, p7_values):
            if maximization and key is 'f':
                p7_value *= -1
            self.assertTrue(np.array_equal(p7ot_value, p7_value))

    def test_Exceptions(self):
        function = ot.NumericalMathFunction(['x1', 'x2'], ['(x1-0.6)^2 + (x2-0.6)^2'])
        # Bounds
        with self.assertRaises(ValueError):
            bounds = ot.Interval([-5]*3, [5]*3)
            problem = ot.OptimizationProblem()
            problem.setObjective(function)
            problem.setBounds(bounds)
            solver = GTOpt(problem)
            solver.setStartingPoint([0, 0])
            solver.run()
        # Initial guess
        with self.assertRaises(ValueError):
            bounds = ot.Interval([-5]*2, [5]*2)
            problem = ot.OptimizationProblem()
            problem.setObjective(function)
            problem.setBounds(bounds)
            solver = GTOpt(problem)
            solver.setStartingPoint([0, 0, 0])
            solver.run()
        # Constraints
        with self.assertRaises(ValueError):
            bounds = ot.Interval([-5]*2, [5]*2)
            problem = ot.OptimizationProblem()
            problem.setObjective(function)
            problem.setEqualityConstraint(ot.NumericalMathFunction(['x1', 'x2', 'x3'], ['x1-x2']))
            problem.setBounds(bounds)
            solver = GTOpt(problem)
            solver.run()
        # Hints
        bounds = ot.Interval([-5]*2, [5]*2)
        problem = ot.OptimizationProblem()
        problem.setObjective(function)
        problem.setBounds(bounds)
        with self.assertRaises(TypeError):
            solver = GTOpt(problem, input_hints={})
        with self.assertRaises(ValueError):
            solver = GTOpt(problem, input_hints=[{}])
        with self.assertRaises(Exception):
            solver = GTOpt(problem, input_hints=[{"@GTOpt/VariableType": "int"}, {}])
            solver.run()
        with self.assertRaises(TypeError):
            solver = GTOpt(problem, input_hints=[{"@GTOpt/VariableType": "integer"}, None])
            solver.run()
        with self.assertRaises(ValueError):
            solver = GTOpt(problem, input_hints=[{"@GTOpt/VariableType": "integer"}, {}])
            solver.setInputHints({"@GTOpt/VariableType": "integer"}, [10])
            solver.run()
        solver = GTOpt(problem, input_hints=[{"@GTOpt/VariableType": "integer"}, {}])
        solver.setInputHints({"@GTOpt/VariableType": "continuous"}, [0])

    def test_Maximization(self):
        dim = 2
        bounds = ot.Interval([-500]*dim, [500]*dim)
        # Objective function
        objective = ot.NumericalMathFunction(['x1', 'x2'], ['- (x1+2*x2-7)^2 - (2*x1+x2-5)^2 + 1'])
        # Define optimization problem
        problem = ot.OptimizationProblem()
        problem.setObjective(objective)
        problem.setBounds(bounds)
        # Maximizaion problem
        problem.setMinimization(False)
        # Prepare solver
        solver = GTOpt(problem)
        solver.setStartingPoint(bounds.getUpperBound())
        # Run optimization and get result
        solver.run()
        result = solver.getResult()
        # Asserts
        outputs_dim = objective.getOutputDimension()
        self.assertEqual(result.getOptimalValue().getDimension(), outputs_dim)
        self.assertEqual(dim + outputs_dim, solver.getP7History().getDimension())
        self.assertEqual(result.getIterationNumber(), solver.getP7History().getSize())
        # Compare the results with those of p7core gtopt

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                p7_bounds = zip(bounds.getLowerBound(), bounds.getUpperBound())
                self.add_variable(bounds=p7_bounds[0], initial_guess=solver.getStartingPoint()[0])
                self.add_variable(bounds=p7_bounds[1], initial_guess=solver.getStartingPoint()[1])
                self.add_objective()

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    functions_batch.append([-1 * value for value in objective(x)])
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        self.compare_results(p7ot_result=solver.getP7Result(), problem=Problem(), maximization=True)

    def test_Rosenbrock(self):
        dim = 4
        bounds = ot.Interval([-500]*dim, [500]*dim)
        inputs = ['x' + str(i) for i in range(dim)]
        formulas = ['']
        for i in range(dim - 1):
            formulas[0] += '(1-x%d)^2+100*(x%d-x%d^2)^2%s' % (i, i+1, i, '' if i == dim - 2 else '+')
        # Objective function
        objective = ot.NumericalMathFunction(inputs, formulas)
        # Define optimization problem
        problem = ot.OptimizationProblem()
        problem.setObjective(objective)
        problem.setBounds(bounds)
        # Prepare solver
        solver = GTOpt(problem)
        solver.setStartingPoint(bounds.getUpperBound())
        # Run optimization and get result
        solver.run()
        result = solver.getResult()
        # Asserts
        outputs_dim = objective.getOutputDimension()
        self.assertEqual(result.getOptimalValue().getDimension(), outputs_dim)
        self.assertEqual(dim + outputs_dim, solver.getP7History().getDimension())
        self.assertEqual(result.getIterationNumber(), solver.getP7History().getSize())
        # Compare the results with those of p7core gtopt

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                p7_bounds = zip(bounds.getLowerBound(), bounds.getUpperBound())
                for i in range(dim):
                    self.add_variable(bounds=p7_bounds[i], initial_guess=solver.getStartingPoint()[i])
                self.add_objective()

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    functions_batch.append(list(objective(x)))
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        self.compare_results(p7ot_result=solver.getP7Result(), problem=Problem())

    def test_LevelFunction(self):
        dim = 4
        bounds = ot.Interval([-500]*dim, [500]*dim)
        # Objective function
        level_function = ot.NumericalMathFunction(['x1', 'x2', 'x3', 'x4'], ['x1+2*x2-3*x3+4*x4'])
        level_value = 3
        # Define optimization problem
        problem = ot.OptimizationProblem()
        problem.setLevelFunction(level_function)
        problem.setLevelValue(level_value)
        problem.setBounds(bounds)
        # Prepare solver
        solver = GTOpt(problem)
        solver.setStartingPoint(bounds.getUpperBound())
        solver.enableConstraintsGradient()
        solver.enableObjectivesGradient()
        # Run optimization and get result
        solver.run()
        result = solver.getResult()
        # Asserts
        outputs_dim = level_function.getOutputDimension() + 1
        gradients_dim = outputs_dim * dim
        self.assertEqual(result.getOptimalValue().getDimension(), outputs_dim - 1)
        self.assertEqual(dim + outputs_dim + gradients_dim, solver.getP7History().getDimension())
        self.assertEqual(result.getIterationNumber(), solver.getP7History().getSize())
        # Compare the results with those of p7core gtopt

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                p7_bounds = zip(bounds.getLowerBound(), bounds.getUpperBound())
                for i in range(dim):
                    self.add_variable(bounds=p7_bounds[i], initial_guess=solver.getStartingPoint()[i])
                self.add_objective()
                self.add_constraint(bounds=(0.0, 0.0))
                self.enable_objectives_gradient()
                self.enable_constraints_gradient()

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    objectives = list(problem.getObjective()(x))
                    constraints = list(problem.getEqualityConstraint()(x))
                    # Objectives gradient
                    objectives_gradient = problem.getObjective().gradient(x)
                    objectives_gradient = sum(np.matrix(objectives_gradient.transpose()).tolist(), [])
                    # Constraints gradient
                    constraints_gradient = problem.getEqualityConstraint().gradient(x)
                    constraints_gradient = sum(np.matrix(constraints_gradient.transpose()).tolist(), [])
                    functions_batch.append(objectives + constraints + objectives_gradient + constraints_gradient)
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        self.compare_results(p7ot_result=solver.getP7Result(), problem=Problem(), level_value=level_value)

    def test_Multiobjective(self):
        dim = 4
        bounds = ot.Interval([-500]*dim, [500]*dim)
        inputs = ['x' + str(i) for i in range(dim)]
        # Rosenbrock function
        formulas = ['']
        for i in range(dim - 1):
            formulas[0] += '(1-x%d)^2+100*(x%d-x%d^2)^2%s' % (i, i+1, i, '' if i == dim - 2 else '+')
        rosenbrock = ot.NumericalMathFunction(inputs, formulas)
        # Sphere function
        formulas = ['']
        for i in range(dim):
            formulas[0] += 'x%d^2%s' % (i, '' if i == dim - 1 else '+')
        sphere = ot.NumericalMathFunction(inputs, formulas)
        # Objective function
        objective = ot.NumericalMathFunction([rosenbrock, sphere])
        # Define optimization problem
        problem = ot.OptimizationProblem()
        problem.setObjective(objective)
        problem.setBounds(bounds)
        # Prepare solver
        solver = GTOpt(problem)
        solver.setStartingPoint(bounds.getUpperBound())
        # Run optimization and get result
        solver.run()
        result = solver.getResult()
        # Asserts
        outputs_dim = objective.getOutputDimension()
        self.assertEqual(result.getOptimalValue().getDimension(), outputs_dim)
        self.assertEqual(dim + outputs_dim, solver.getP7History().getDimension())
        self.assertEqual(result.getIterationNumber(), solver.getP7History().getSize())
        # Compare the results with those of p7core gtopt

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                p7_bounds = zip(bounds.getLowerBound(), bounds.getUpperBound())
                for i in range(dim):
                    self.add_variable(bounds=p7_bounds[i], initial_guess=solver.getStartingPoint()[i])
                self.add_objective()
                self.add_objective()

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    functions_batch.append(list(objective(x)))
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        self.compare_results(p7ot_result=solver.getP7Result(), problem=Problem())

    def test_MultiobjectiveWithConstraints(self):
        dim = 4
        bounds = ot.Interval([-500]*dim, [500]*dim)
        inputs = ['x' + str(i) for i in range(dim)]
        # Rosenbrock function
        formulas = ['']
        for i in range(dim - 1):
            formulas[0] += '(1-x%d)^2+100*(x%d-x%d^2)^2%s' % (i, i+1, i, '' if i == dim - 2 else '+')
        rosenbrock = ot.NumericalMathFunction(inputs, formulas)
        # Sphere function
        formulas = ['']
        for i in range(dim):
            formulas[0] += 'x%d^2%s' % (i, '' if i == dim - 1 else '+')
        sphere = ot.NumericalMathFunction(inputs, formulas)
        # Objective function
        objective = ot.NumericalMathFunction([rosenbrock, sphere])
        # Equality constraint
        equality = ot.NumericalMathFunction(inputs, ['x0-3*x%d' % (dim-1)])
        # Define optimization problem
        problem = ot.OptimizationProblem()
        problem.setObjective(objective)
        problem.setEqualityConstraint(equality)
        problem.setBounds(bounds)
        # Prepare solver
        solver = GTOpt(problem)
        solver.setStartingPoint(bounds.getUpperBound())
        # Run optimization and get result
        solver.run()
        result = solver.getResult()
        # Asserts
        outputs_dim = objective.getOutputDimension() + equality.getOutputDimension()
        self.assertEqual(result.getOptimalValue().getDimension(), outputs_dim)
        self.assertEqual(dim + outputs_dim, solver.getP7History().getDimension())
        self.assertEqual(result.getIterationNumber(), solver.getP7History().getSize())
        # Compare the results with those of p7core gtopt

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                p7_bounds = zip(bounds.getLowerBound(), bounds.getUpperBound())
                for i in range(dim):
                    self.add_variable(bounds=p7_bounds[i], initial_guess=solver.getStartingPoint()[i])
                self.add_objective()
                self.add_objective()
                self.add_constraint(bounds=(0.0, 0.0))

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    functions_batch.append(list(objective(x)) + list(equality(x)))
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        self.compare_results(p7ot_result=solver.getP7Result(), problem=Problem())

    def test_MultiobjectiveWithConstraintsAndGradients(self):
        dim = 4
        bounds = ot.Interval([-500]*dim, [500]*dim)
        inputs = ['x' + str(i) for i in range(dim)]
        # Rosenbrock function
        formulas = ['']
        for i in range(dim - 1):
            formulas[0] += '(1-x%d)^2+100*(x%d-x%d^2)^2%s' % (i, i+1, i, '' if i == dim - 2 else '+')
        rosenbrock = ot.NumericalMathFunction(inputs, formulas)
        # Sphere function
        formulas = ['']
        for i in range(dim):
            formulas[0] += 'x%d^2%s' % (i, '' if i == dim - 1 else '+')
        sphere = ot.NumericalMathFunction(inputs, formulas)
        # Objective function
        objective = ot.NumericalMathFunction([rosenbrock, sphere])
        # Inequality constraint
        inequality = ot.NumericalMathFunction(inputs, ['x0-x%d' % (dim-1)])
        # Define optimization problem
        problem = ot.OptimizationProblem()
        problem.setObjective(objective)
        problem.setEqualityConstraint(inequality)
        problem.setBounds(bounds)
        # Prepare solver
        solver = GTOpt(problem)
        solver.setStartingPoint(bounds.getUpperBound())
        solver.enableConstraintsGradient()
        solver.enableObjectivesGradient()
        # Run optimization and get result
        solver.run()
        result = solver.getResult()
        # Asserts
        outputs_dim = objective.getOutputDimension() + inequality.getOutputDimension()
        gradients_dim = outputs_dim * dim
        self.assertEqual(result.getOptimalValue().getDimension(), outputs_dim)
        self.assertEqual(dim + outputs_dim + gradients_dim, solver.getP7History().getDimension())
        self.assertEqual(result.getIterationNumber(), solver.getP7History().getSize())
        # Compare the results with those of p7core gtopt

        class Problem(gtopt.ProblemGeneric):

            def prepare_problem(self):
                p7_bounds = zip(bounds.getLowerBound(), bounds.getUpperBound())
                for i in range(dim):
                    self.add_variable(bounds=p7_bounds[i], initial_guess=solver.getStartingPoint()[i])
                self.add_objective()
                self.add_objective()
                self.add_constraint(bounds=(0.0, 0.0))
                self.enable_objectives_gradient()
                self.enable_constraints_gradient()

            def evaluate(self, queryx, querymask):
                functions_batch = []
                output_masks_batch = []
                for x, mask in zip(queryx, querymask):
                    objectives = list(objective(x))
                    constraints = list(inequality(x))
                    # Objectives gradient
                    objectives_gradient = problem.getObjective().gradient(x)
                    objectives_gradient = sum(np.matrix(objectives_gradient.transpose()).tolist(), [])
                    # Constraints gradient
                    constraints_gradient = problem.getEqualityConstraint().gradient(x)
                    constraints_gradient = sum(np.matrix(constraints_gradient.transpose()).tolist(), [])
                    functions_batch.append(objectives + constraints + objectives_gradient + constraints_gradient)
                    output_masks_batch.append(mask)
                return functions_batch, output_masks_batch

        self.compare_results(p7ot_result=solver.getP7Result(), problem=Problem())

if __name__ == '__main__':
    unittest.main()
