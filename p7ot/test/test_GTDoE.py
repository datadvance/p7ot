import openturns as ot
from da.p7core import gtdoe, blackbox
import unittest
import numpy as np
from ..gtdoe import (
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


class TestGTDoE(unittest.TestCase):

    def compare_results(self, p7ot_result, p7_result):
        p7ot_result = np.array(p7ot_result)
        p7_result = p7_result.points
        self.assertTrue(np.array_equal(p7ot_result, p7_result))

    def convert_bounds(self, p7ot_bounds):
        return (np.array(p7ot_bounds.getLowerBound()), np.array(p7ot_bounds.getUpperBound()))

    def test_Sequence(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        generator = Sequence(bounds=bounds, count=count)
        generator.setDeterministic(True)
        self.assertEqual(generator.getTechnique(), 'SobolSeq')
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        self.assertEqual(result.getSize(), count)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertTrue(technique in ["RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"])
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': technique, 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, options=p7_options)
        self.compare_results(result, p7_result)
        # Exceptions
        with self.assertRaises(TypeError):
            # Invalid bounds
            Sequence(bounds=10, count=10)
        with self.assertRaises(TypeError):
            # Invalid count
            Sequence(bounds=ot.Interval([10, 4], [5, 10]), count=None).generate()
        with self.assertRaises(ValueError):
            # Invalid count
            Sequence(bounds=ot.Interval([10, 4], [5, 10]), count=0).generate()
        with self.assertRaises(TypeError):
            # Invalid technique
            exp = Sequence(bounds=ot.Interval([10, 4], [5, 10]), count=10, technique=10)
        with self.assertRaises(ValueError):
            # Invalid technique
            exp = Sequence(bounds=ot.Interval([10, 4], [5, 10]), count=10, technique='Random')

    def test_LHS(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        generator = LHS(bounds=bounds, count=count)
        # Default technique is LHS (useOptimized=False)
        self.assertEqual(generator._settings.get('technique'), 'LHS')
        generator.enableOptimized()
        generator.setDeterministic(True)
        self.assertEqual(generator._settings.get('technique'), 'OLHS')
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        self.assertEqual(result.getSize(), count)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertTrue(technique in ["LHS", "OLHS"])
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': technique, 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, options=p7_options)
        self.compare_results(result, p7_result)
        # Exceptions
        with self.assertRaises(TypeError):
            # Invalid bounds
            LHS(bounds=10, count=10)
        with self.assertRaises(TypeError):
            # Invalid count
            LHS(bounds=ot.Interval([10, 4], [5, 10]), count=None).generate()
        with self.assertRaises(ValueError):
            # Invalid count
            LHS(bounds=ot.Interval([10, 4], [5, 10]), count=0).generate()

    def test_BoxBehnken(self):
        bounds = ot.Interval([-10]*3, [10]*3)
        dim = bounds.getDimension()
        count = dim * 2
        generator = BoxBehnken(bounds=bounds, count=count)
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        # The number of points should be in range [1, 2d(d-1)+1]
        self.assertTrue(result.getSize() <= 2 * dim * (dim - 1) + 1)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "BoxBehnken")
        # Specific case for Box-Behnken design of experiments when count is None
        generator = BoxBehnken(bounds=bounds, count=None)
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(generator.getCount(), 2 * dim * (dim - 1) + 1)
        self.assertEqual(result.getSize(), 2 * dim * (dim - 1) + 1)
        # Exceptions
        with self.assertRaises(Exception):
            # Invalid count, should be in range [1, 2d(d-1)+1]
            BoxBehnken(bounds=bounds, count=dim*5).generate()

    def test_FullFactorial(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        dim = bounds.getDimension()
        generator = FullFactorial(bounds=bounds, count=count)
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        # Actual number of points to be generated
        self.assertEqual(result.getSize(), int(count**(1.0/dim))**dim)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "FullFactorial")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': "FullFactorial", 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, options=p7_options)
        self.compare_results(result, p7_result)
        # Exceptions
        with self.assertRaises(Exception):
            # Invalid count, should be in range [2^dim, 2^31-2]
            FullFactorial(bounds=bounds, count=dim).generate()

    def test_FractionalFactorial(self):
        count = 4
        bounds = ot.Interval([-10]*3, [10]*3)
        dim = bounds.getDimension()
        generator = FractionalFactorial(bounds=bounds, count=count,
                                        mainFactors=[0, 2], generatingString='a ab b')
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        # The number of points should be in range [1, 2^d]
        self.assertTrue(result.getSize() <= 2**dim)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "FractionalFactorial")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': "FractionalFactorial", 'GTDoE/Deterministic': True,
                      'GTDoE/FractionalFactorial/GeneratingString': 'a ab b',
                      'GTDoE/FractionalFactorial/MainFactors': [0, 2]}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, options=p7_options)
        self.compare_results(result, p7_result)
        # Exceptions
        with self.assertRaises(Exception):
            # Invalid count
            FractionalFactorial(bounds=bounds, count=dim).generate()
        with self.assertRaises(Exception):
            # Invalid count
            FractionalFactorial(bounds=bounds, count=3**dim).generate()

    def test_OptimalDesign(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        generator = OptimalDesign(bounds=bounds, count=count, model="quadratic")
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        # The number of points should be in range [1, count].
        # Actual number of points depends on the value of model parameter.
        self.assertTrue(result.getSize() <= count)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "OptimalDesign")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': "OptimalDesign", 'GTDoE/Deterministic': True,
                      'GTDoE/OptimalDesign/Model': 'quadratic'}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, options=p7_options)
        self.compare_results(result, p7_result)

    def test_OrthogonalArray(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        generator = OrthogonalArray(bounds=bounds, levelsNumber=[2, 3, 4])
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        # count value depends on levelsNumber
        self.assertEqual(result.getSize(), generator.getCount())
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "OrthogonalArray")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': 'OrthogonalArray', 'GTDoE/Deterministic': True,
                      'GTDoE/OrthogonalArray/LevelsNumber': [2, 3, 4]}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=generator.getCount(), options=p7_options)
        self.compare_results(result, p7_result)
        # Exceptions
        with self.assertRaises(Exception):
            # Count should be less than 150
            OrthogonalArray(bounds=bounds, levelsNumber=[20, 3, 4]).generate()

    def test_ParametricStudy(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        generator = ParametricStudy(bounds=bounds, count=count)
        generator.setDeterministic(True)
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        self.assertEqual(result.getSize(), count)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "ParametricStudy")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': 'ParametricStudy', 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, options=p7_options)
        self.compare_results(result, p7_result)

    def test_AdaptiveBlackbox(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        function = ot.NumericalMathFunction(['x1', 'x2', 'x3'], ['x1 + 2*x2 + 3*x3'])
        generator = AdaptiveBlackbox(bounds=bounds, count=count, blackbox=function)
        generator.setDeterministic(True)
        self.assertEqual(count, generator._settings.get('budget'))
        self.assertEqual(generator.getCount(), generator._settings.get('budget'))
        result = generator.generate()
        self.assertEqual(result.getDimension(), bounds.getDimension())
        self.assertEqual(result.getSize(), count)
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "Adaptive")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': "Adaptive", 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)

        class P7_Blackbox(blackbox.Blackbox):

            def prepare_blackbox(self):
                # Add variables
                for i in range(function.getInputDimension()):
                    self.add_variable(bounds=(p7_bounds[0][i], p7_bounds[1][i]), name=function.getInputDescription()[i])
                # Add responses
                self.add_response(name=function.getOutputDescription()[0])

            def evaluate(self, points):
                result = []
                for point in points:
                    result.append(list(function(point)))
                return result

        p7_blackbox = P7_Blackbox()
        p7_result = gtdoe.Generator().generate(blackbox=p7_blackbox, bounds=p7_bounds, budget=count, options=p7_options)
        self.compare_results(result, p7_result)
        # Exceptions
        with self.assertRaises(TypeError):
            # Invalid blackbox
            AdaptiveBlackbox(bounds=bounds, count=count, blackbox=None).generate()
        with self.assertRaises(ValueError):
            # Inconsistent blackbox and bounds dimension
            AdaptiveBlackbox(bounds=ot.Interval([1, 4], [5, 10]), count=count, blackbox=function).generate()

    def test_AdaptiveSample(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        init_x = np.random.random((20, 3))
        generator = AdaptiveSample(bounds=bounds, count=count, init_x=init_x)
        generator.setDeterministic(True)
        result = generator.generate()
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "Adaptive")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': "Adaptive", 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, init_x=init_x, options=p7_options)
        self.compare_results(result, p7_result)
        # In case of specified init_y
        init_y = [np.sum(x)*np.sum(x) for x in init_x]
        generator = AdaptiveSample(bounds=bounds, count=count, init_x=init_x, init_y=init_y)
        generator.setCriterion('IntegratedMseGainMaxVar')
        generator.setDeterministic(True)
        result = generator.generate()
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertEqual(technique, "Adaptive")
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': "Adaptive", 'GTDoE/Deterministic': True,
                      'GTDoE/Adaptive/Criterion': 'IntegratedMseGainMaxVar'}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, init_x=init_x,
                                               init_y=init_y, options=p7_options)
        self.compare_results(result, p7_result)
        with self.assertRaises(Exception):
            # Wrong initial sample
            generator = AdaptiveSample(bounds=bounds, count=count, init_x=bounds, criterion='IntegratedMseGainMaxVar')
            generator.generate()
        with self.assertRaises(Exception):
            # Criterion can't be set without init_y
            generator = AdaptiveSample(bounds=bounds, count=count, init_x=init_x, criterion='IntegratedMseGainMaxVar')
            generator.generate()
        with self.assertRaises(Exception):
            # Inconsistent initial set and bounds dimension
            generator = AdaptiveSample(bounds=ot.Interval([-10]*3, [10]*3), count=count, init_x=[1, 2]*10)
            generator.generate()

    def test_AdaptiveLHS(self):
        count = 20
        bounds = ot.Interval([-10]*3, [10]*3)
        init_x = np.random.random((20, 3))
        generator = AdaptiveLHS(bounds=bounds, count=count, init_x=init_x, useOptimized=True)
        # Default technique is LHS (useOptimized=False)
        self.assertEqual(generator._settings.get('technique'), 'OLHS')
        generator.disableOptimized()
        generator.setDeterministic(True)
        self.assertEqual(generator._settings.get('technique'), 'LHS')
        result = generator.generate()
        technique = generator.getP7Result().info["Generator"]["Technique"]
        self.assertTrue(technique in ["LHS", "OLHS"])
        # Compare the results with those of p7core generator
        p7_options = {'GTDoE/Technique': technique, 'GTDoE/Deterministic': True}
        p7_bounds = self.convert_bounds(bounds)
        p7_result = gtdoe.Generator().generate(bounds=p7_bounds, count=count, init_x=init_x, options=p7_options)
        self.compare_results(result, p7_result)

if __name__ == '__main__':
    unittest.main()
