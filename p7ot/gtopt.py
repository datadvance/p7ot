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
from da.p7core import gtopt
from .problem import _ProblemGeneric
import numpy as np


class _HintsList(object):
    def __init__(self, hints_list, dimension):
        # Check arguments and create valid list of hints that are to be used to prepare p7 problem.
        if hints_list is None:
            hints_list = [{}] * dimension
        else:
            # hints_list should be a list
            if not isinstance(hints_list, list):
                raise TypeError("Wrong type of hints list. Expected: list of dictionaries, got: %s" % type(hints_list))
            # Check the length of the hints_list
            if len(hints_list) != dimension:
                raise ValueError("Wrong length of hints list. Expected: %s, got: %s" % (dimension, len(hints_list)))
            # Elements of the hints_list should be a dictionary
            invalid_hints = [type(item) for item in hints_list if not isinstance(item, dict)]
            if invalid_hints:
                raise TypeError("Wrong type of hints. Expected: dictionary, got: %s" % invalid_hints)
        self.__hints_list = hints_list
        self.__dimension = dimension

    def apply(self, hints, indices=None):
        # hints should be a dictionary
        if not isinstance(hints, dict):
            raise TypeError("Wrong type of hints. Expected: dictionary, got: %s" % type(hints))
        # Check indices
        indices = self.__validate_indices(indices)
        # Modify the list of hints
        for i in indices:
            self.__hints_list[i] = hints

    def get_list(self):
        return self.__hints_list

    def __validate_indices(self, indices=None):
        # Check indicies if specified (should be a list of integers or a single integer)
        if indices is None:
            # hints will be set to each element of the hints list
            indices = range(self.__dimension)
        elif isinstance(indices, int):
            # In case of getting a single index
            if indices not in range(self.__dimension):
                raise ValueError("The index must be in range 0...%s, got %s" % (self.__dimension-1, indices))
            indices = [indices]
        elif isinstance(indices, list):
            # Remove dublicates from the indicies list
            indices = list(set(indices))
            # Indices should be in valid range [0, dimension-1]
            invalid_indices = [item for item in indices if item not in range(self.__dimension)]
            if invalid_indices:
                raise ValueError("The indices must be in range 0...%s, got %s" % (self.__dimension-1, invalid_indices))
        else:
            raise TypeError("Wrong type of indices. Expected: list or integer, got: %s" % (type(indices)))
        return indices


class GTOpt(ot.OptimizationSolverImplementation):
    """
    Generic Tool for Optimization (GTOpt) module from pSeven Core.

    Available constructors:
        GTOpt(*problem, options=None,
               input_hints=None, objective_hints=None, equality_hints=None, inequality_hints=None,
               sample_x=None, sample_f=None, sample_c=None*)

    Parameters
    ----------
    problem: :class:`~openturns.OptimizationProblem`
        Optimization problem.
    options: dictionary
        Solver options.
    input_hints: list of dictionaries
        Additional properties of defined design variables.
    objective_hints: list of dictionaries
        Additional properties of defined objective functions.
    equality_hints: list of dictionaries
        Additional properties of defined equality constraints.
    inequality_hints: list of dictionaries
        Additional properties of defined inequality constraints.
    sample_x: array-like, 1D or 2D
        Optional initial sample containing values of variables.
    sample_f: array-like, 1D or 2D
        Optional initial sample of objective function values, requires sample_x.
    sample_c: array-like, 1D or 2D
        Optional initial sample of constraint function values, requires sample_x.
    """
    def __init__(self, problem, options=None,
                 input_hints=None, objective_hints=None, equality_hints=None, inequality_hints=None,
                 sample_x=None, sample_f=None, sample_c=None):
        super(GTOpt, self).__init__(problem)
        self.__p7_history = None
        self.__p7_problem = None
        self.__p7_result = None
        self.__ot_result = None
        self.__options = gtopt.Solver().options
        if options:
            self.__options.set(options)
        self.__sample_x = sample_x
        self.__sample_f = sample_f
        self.__sample_c = sample_c
        self.__use_objectives_gradient = False
        self.__use_constraints_gradient = False
        # Dimentions
        self.__input_dim = problem.getDimension()
        self.__objectives_dim = problem.getObjective().getOutputDimension()
        self.__equality_dim = problem.getEqualityConstraint().getOutputDimension()
        self.__inequality_dim = problem.getInequalityConstraint().getOutputDimension()
        # GTOpt hints
        self.__input_hints = _HintsList(input_hints, self.__input_dim)
        self.__objective_hints = _HintsList(objective_hints, self.__objectives_dim)
        self.__equality_hints = _HintsList(equality_hints, self.__equality_dim)
        self.__inequality_hints = _HintsList(inequality_hints, self.__inequality_dim)

    def getClassName(self):
        """
        Accessor to the object's name.

        Returns
        -------
        class_name: str
            The object class name (`object.__class__.__name__`).
        """
        return self.__class__.__name__

    def getP7Problem(self):
        """
        Accessor to p7core optimization problem.

        Returns
        -------
        p7_problem: :class:`~da.p7core.gtopt.ProblemGeneric`
            Problem object.
        """
        return self.__p7_problem

    def getP7Result(self):
        """
        Accessor to p7core optimization result.

        Returns
        -------
        p7result: :class:`~da.p7core.gtopt.Result`
            Result object.
        """
        return self.__p7_result

    def getP7History(self):
        """
        Accessor to the history of problem evaluations.

        Returns
        -------
        p7result: :class:`~openturns.NumericalSample`
            Returns values of variables and evaluation results.
            Each element of the top-level list is one evaluated point.
            Nested list structure is [variables, objectives, constraints, objective gradients, constraint gradients].
            Gradients are added only if analytical gradients are enabled.
        """
        return ot.NumericalSample(self.__p7_history)

    def getResult(self):
        """
        Accessor to optimization result.

        Returns
        -------
        result: :class:`~openturns.OptimizationResult`
            Result class.
        """
        return self.__ot_result

    def setResult(self, result):
        """
        Accessor to optimization result.

        Parameters
        ----------
        result: :class:`~openturns.OptimizationResult`
            Result class.
        """
        self.__ot_result = result

    def run(self):
        """Launch the optimization."""
        self.__p7_problem = _ProblemGeneric(problem=self.getProblem(),
                                            starting_point=self.getStartingPoint(),
                                            use_objectives_gradient=self.__use_objectives_gradient,
                                            use_constraints_gradient=self.__use_constraints_gradient,
                                            input_hints=self.__input_hints.get_list(),
                                            objective_hints=self.__objective_hints.get_list(),
                                            equality_hints=self.__equality_hints.get_list(),
                                            inequality_hints=self.__inequality_hints.get_list())
        self.__p7_problem.set_history(memory=True)
        self.__p7_result = gtopt.Solver().solve(problem=self.__p7_problem, options=self.__options.get(),
                                                sample_x=self.__sample_x, sample_f=self.__sample_f,
                                                sample_c=self.__sample_c)
        # Convert None values to nan in history
        self.__p7_history = np.array(self.__p7_problem.designs, dtype=np.float)
        # In case of maximization problem
        if not self.getProblem().isMinimization():
                self.__p7_history[:, self.__input_dim: self.__input_dim + self.__objectives_dim] *= -1
        # Convert p7 result to openturns result
        self.__ot_result = self.__get_ot_result(iteration_number=len(self.__p7_history))

    def getMaximumIterationNumber(self):
        """
        Get the maximum allowed number of iterations.

        Returns
        -------
        maximumIterationNumber: int
            Maximum allowed number of iterations.
        """
        # Default value is 0
        self.__options.get('GTOpt/MaximumIterations')

    def setMaximumIterationNumber(self, value):
        """
        Set the maximum allowed number of iterations.

        Parameters
        ----------
        maximumIterationNumber: int
            Maximum allowed number of iterations.
        """
        self.__options.set('GTOpt/MaximumIterations', value)

    def getVerbose(self):
        """
        Get the verbosity flag.

        Returns
        -------
        verbose: bool
            Verbosity flag state.
        """
        # Default value is False
        self.__options.get('GTOpt/VerboseOutput')

    def setVerbose(self, value):
        """
        Set the verbosity flag.

        Parameters
        ----------
        verbose: bool
            Verbosity flag state.
        """
        self.__options.set('GTOpt/VerboseOutput', value)

    def enableObjectivesGradient(self):
        """Enable the use of analytical gradients of objective functions."""
        self.__use_objectives_gradient = True

    def disableObjectivesGradient(self):
        """Disable the use of analytical gradients of objective functions."""
        self.__use_objectives_gradient = False

    def isObjectivesGradientEnabled(self):
        """
        Test whether the analytical gradients of objective functions are enabled or not.

        Returns
        -------
        useObjectivesGradient: bool
            Flag telling whether the analytical gradients of objective functions are enabled (Default: False).
        """
        return self.__use_objectives_gradient

    def enableConstraintsGradient(self):
        """Enable the use of analytical gradients of constraint functions."""
        self.__use_constraints_gradient = True

    def disableConstraintsGradient(self):
        """Disable the use of analytical gradients of constraint functions."""
        self.__use_constraints_gradient = False

    def isConstraintsGradientEnabled(self):
        """
        Test whether the analytical gradients of constraint functions are enabled or not.

        Returns
        -------
        useConstraintGradient: bool
            Flag telling whether the analytical gradients of constraint functions are enabled (Default: False)
        """
        return self.__use_constraints_gradient

    def getInputHints(self):
        """
        Get the properties of defined design variables.

        Returns
        -------
        hints: dictionary
            Hints (dictionary keys) and their values.
        """
        return self.__input_hints.get_list()

    def setInputHints(self, hints, indices=None):
        """
        Set the properties of defined design variables.

        Parameters
        ----------
        hints: dictionary
            Hints (dictionary keys) and their values.
        indices: list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__input_hints.apply(hints=hints, indices=indices)

    def getObjectiveHints(self):
        """
        Get the properties of defined objective functions.

        Returns
        -------
        hints: dictionary
            Hints (dictionary keys) and their values.
        """
        return self.__objective_hints.get_list()

    def setObjectiveHints(self, hints, indices=None):
        """
        Set the properties of defined objective functions.

        Parameters
        ----------
        hints: dictionary
            Hints (dictionary keys) and their values.
        indices: list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__objective_hints.apply(hints=hints, indices=indices)

    def getEqualityConstraintHints(self):
        """
        Get the properties of defined equality constraints.

        Returns
        -------
        hints: dictionary
            Hints (dictionary keys) and their values.
        """
        return self.__equality_hints.get_list()

    def setEqualityConstraintHints(self, hints, indices=None):
        """
        Set the properties of defined equality constraints.

        Parameters
        ----------
        hints: dictionary
            Hints (dictionary keys) and their values.
        indices: list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__equality_hints.apply(hints=hints, indices=indices)

    def getInequalityConstraintHints(self):
        """
        Get the properties of defined inequality constraints.

        Returns
        -------
        hints: dictionary
            Hints (dictionary keys) and their values.
        """
        return self.__inequality_hints.get_list()

    def setInequalityConstraintHints(self, hints, indices=None):
        """
        Set the properties of defined inequality constraints.

        Parameters
        ----------
        hints: dictionary
            Hints (dictionary keys) and their values.
        indices: list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__inequality_hints.apply(hints=hints, indices=indices)

    def __get_ot_result(self, iteration_number):
        optimal_x = self.__p7_result.optimal.x
        optimal_f = self.__p7_result.optimal.f
        optimal_c = self.__p7_result.optimal.c
        # Convert values in case of maximization problem (p7core.gtopt doesn't support it)
        if not self.getProblem().isMinimization():
            optimal_f *= -1
        # In case of optimization problem with level function g(x)=v:
        # 1. level function g(x)=v acts as equality constraint, min||x|| - as objective function
        # 2. the value of level function should be set as optimal
        optimal_outputs = None
        if self.getProblem().hasLevelFunction():
            level_value = self.getProblem().getLevelValue()
            optimal_outputs = [optimal_c[0] + level_value]
        else:
            optimal_outputs = np.concatenate((optimal_f, optimal_c), axis=1)
        # Create optimization result object
        # Absolute, constraint, residual and relative errors are not supported by p7 builder.
        # For additional information about the solving process see getP7Result() and getP7History()
        ot_result = ot.OptimizationResult(optimal_x[0], optimal_outputs[0], iteration_number, -1, -1, -1, -1)
        return ot_result
