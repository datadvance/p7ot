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

from da.p7core import gtopt
import numpy as np


class _ProblemGeneric(gtopt.ProblemGeneric):
    def __init__(self, problem, starting_point, use_objectives_gradient, use_constraints_gradient,
                 input_hints, objective_hints, equality_hints, inequality_hints):
        super(_ProblemGeneric, self).__init__()
        # Prepare information required for p7 problem
        # openturns.OptimizationProblem object
        self.ot_problem = problem
        # Objectives, constraints, gradients functions (openturns.NumericalMathFunction objects)
        self.ot_objectives = problem.getObjective()
        self.ot_equality = None
        self.ot_inequality = None
        self.ot_objectives_gradient = None
        self.ot_equality_gradient = None
        self.ot_inequality_gradient = None
        # Variables
        self.variables_initial_guess = starting_point or [None] * self.ot_problem.getDimension()
        self.variables_bounds_ = self.__get_variables_bounds()  # can't be named as variables_bounds
        # GTopt hints
        self.variables_hints = input_hints
        self.objectives_hints = objective_hints
        self.equality_hints = equality_hints
        self.inequality_hints = inequality_hints
        # Dimensions: if gradients are not enabled, the appropriate dimensions should be 0
        self.variables_dim = self.ot_problem.getDimension()
        self.objectives_dim = self.ot_problem.getObjective().getOutputDimension()
        self.equality_dim = self.ot_problem.getEqualityConstraint().getOutputDimension()
        self.inequality_dim = self.ot_problem.getInequalityConstraint().getOutputDimension()
        self.objectives_gradient_dim = self.objectives_dim * self.variables_dim if use_objectives_gradient else 0
        self.equality_gradient_dim = self.equality_dim * self.variables_dim if use_constraints_gradient else 0
        self.inequality_gradient_dim = self.inequality_dim * self.variables_dim if use_constraints_gradient else 0
        # Check constraints
        if problem.hasEqualityConstraint():
            self.ot_equality = problem.getEqualityConstraint()
        if problem.hasInequalityConstraint():
            self.ot_inequality = problem.getInequalityConstraint()
        # Check objectives gradient
        if use_objectives_gradient:
            self.ot_objectives_gradient = self.ot_objectives.getGradient()
        # Check constraints gradient
        if use_constraints_gradient:
            self.ot_equality_gradient = self.ot_equality.getGradient()
            self.ot_inequality_gradient = self.ot_inequality.getGradient()

    def prepare_problem(self):
        # Add variables
        variables_names = self.ot_problem.getObjective().getInputDescription()
        for i in range(self.variables_dim):
            self.add_variable(bounds=self.variables_bounds_[i], initial_guess=self.variables_initial_guess[i],
                              name=variables_names[i], hints=self.variables_hints[i])
        # Add objectives
        objectives_names = self.ot_problem.getObjective().getOutputDescription()
        for i in range(self.objectives_dim):
            self.add_objective(name=objectives_names[i], hints=self.objectives_hints[i])
        # The names of constraints may not be unique as they refferes to different objects
        # Add equality constraints g(x) = 0
        if self.ot_equality is not None:
            for i in range(self.equality_dim):
                self.add_constraint(bounds=(0.0, 0.0), name=None, hints=self.equality_hints[i])
        # Add inequality constraints h(x) > 0
        if self.ot_inequality is not None:
            for i in range(self.inequality_dim):
                self.add_constraint(bounds=(0.0, None), name=None, hints=self.inequality_hints[i])
        # Enable specified gradients
        if self.ot_objectives_gradient is not None:
            self.enable_objectives_gradient()
        if self.ot_equality_gradient is not None and self.ot_inequality_gradient is not None:
            self.enable_constraints_gradient()

    def evaluate(self, queryx, querymask):
        functions_batch = []
        output_masks_batch = []
        for x, mask in zip(queryx, querymask):
            # Variables are used to fill outputs part by part
            first_value_index = 0
            last_value_index = 0
            submask = []
            values = []
            # Prepare objectives
            objectives = [None] * self.objectives_dim
            last_value_index = self.objectives_dim
            submask = mask[first_value_index: last_value_index]
            values = self.__calc_partially(function=self.ot_objectives, x=x, mask=submask)
            # Assign objectives
            objectives = values
            # Prepare constraints
            constraints = [None] * (self.equality_dim + self.inequality_dim)
            if self.ot_equality is not None:
                first_value_index = last_value_index
                last_value_index += self.equality_dim
                submask = mask[first_value_index: last_value_index]
                values = self.__calc_partially(function=self.ot_equality, x=x, mask=submask)
                # Assign equality constraints
                constraints[: self.equality_dim] = list(values)
            if self.ot_inequality is not None:
                first_value_index = last_value_index
                last_value_index += self.inequality_dim
                submask = mask[first_value_index: last_value_index]
                values = self.__calc_partially(function=self.ot_inequality, x=x, mask=submask)
                # Assign inequality constraints
                constraints[self.equality_dim:] = list(values)
            # Prepare objectives gradient
            objectives_gradient = [None] * self.objectives_gradient_dim
            if self.ot_objectives_gradient is not None:
                first_value_index = last_value_index
                last_value_index += self.objectives_gradient_dim
                submask = mask[first_value_index: last_value_index]
                values = self.__calc_gradient_partially(function=self.ot_objectives_gradient, x=x, mask=submask)
                # Assign objectives gradient
                objectives_gradient = values
            # Prepare constraints gradient
            constraints_gradient = [None] * (self.equality_gradient_dim + self.inequality_gradient_dim)
            if self.ot_equality_gradient is not None:
                first_value_index = last_value_index
                last_value_index += self.equality_gradient_dim
                submask = mask[first_value_index: last_value_index]
                values = self.__calc_gradient_partially(function=self.ot_equality_gradient, x=x, mask=submask)
                # Assign equality constraints gradient
                constraints_gradient[: self.equality_gradient_dim] = values
            if self.ot_inequality_gradient is not None:
                first_value_index = last_value_index
                last_value_index += self.inequality_gradient_dim
                submask = mask[first_value_index: last_value_index]
                values = self.__calc_gradient_partially(function=self.ot_inequality_gradient, x=x, mask=submask)
                # Assign inequality constraints gradient
                constraints_gradient[self.equality_gradient_dim:] = values

            # p7core.gtopt doesn't support maximization problem
            if not self.ot_problem.isMinimization():
                objectives = [-1*i if i else None for i in objectives]

            functions_batch.append(objectives + constraints + objectives_gradient + constraints_gradient)
            output_mask = mask
            output_masks_batch.append(output_mask)
        return functions_batch, output_masks_batch

    def __to_lists(self, jacobian):
        # Convert the Jacobian transposed matrix (returned by openturns NumericalMathFunction) to a lists
        return np.matrix(jacobian.transpose()).tolist()

    def __get_variables_bounds(self):
        # Prepares the openturns bounds for p7core solver
        variables_bounds = []
        if self.ot_problem.hasBounds():
            # openturns.Interval object
            ot_bounds = self.ot_problem.getBounds()
            # Parse openturns bounds to get bounds for p7 solver
            for i in range(ot_bounds.getDimension()):
                variables_bounds.append([
                    ot_bounds.getLowerBound()[i] if ot_bounds.getFiniteLowerBound()[i] == 1 else None,
                    ot_bounds.getUpperBound()[i] if ot_bounds.getFiniteUpperBound()[i] == 1 else None
                ])
        else:
            for i in range(self.ot_problem.getObjective().getInputDimension()):
                variables_bounds.append([None, None])
        return variables_bounds

    # Calculate the certain part of outputs using the apropriate function
    def __calc_partially(self, function, x, mask):
        result = [None] * len(mask)
        if all(mask):
            # Calculate and return all values, convert returned ot.NumericalPoint to list
            result = list(function(x))
        elif any(mask):
            # Calculate values partially
            # Fill set of indices for which the marginal function will be extracted [0, 1, 0, 1] -> [1, 3]
            marginal_mask = [i for i, item in enumerate(mask) if item]
            # Extract marginal function and calculate values
            marginal_result = function.getMarginal(marginal_mask)(x)
            # Assign calculated values to result
            for i, result_index in enumerate(marginal_mask):
                result[result_index] = marginal_result[i]
        return result

    # Calculate the certain part of gradient using the apropriate function
    def __calc_gradient_partially(self, function, x, mask):
        # Partially gradient calculation is supported by openturns for outputs only (Jacobian rows)
        result = [None] * len(mask)
        # Fill mask for outputs
        # Calculate the row of Jacobian matrix if any of values of the mask for this row is 1
        outputs_mask = [1 if any(mask[i: i+len(x)]) else 0 for i in range(0, len(mask), len(x))]
        if all(outputs_mask):
            # Calculate and return all values as 1D list
            result = sum(self.__to_lists(function.gradient(x)), [])
        elif any(mask):
            # Calculate values partially
            # Fill set of indices for which the marginal function will be extracted [0, 1, 0, 1] -> [1, 3]
            marginal_mask = [i for i, item in enumerate(outputs_mask) if item]
            # Extract marginal function and calculate values
            # marginal_mask - indices of functions to be calculated (rows of Jacodian matrix)
            marginal_function = function.getMarginal(marginal_mask)
            # Calculate required rows of Jacobian matrix and convert result to lists
            marginal_result = self.__to_lists(marginal_function.gradient(x))
            # Assign calculated values to result
            for i, result_index in enumerate(marginal_mask):
                row_first_index = result_index * len(x)
                row_last_index = row_first_index + len(x)
                result[row_first_index: row_last_index] = marginal_result[i]
        return result
