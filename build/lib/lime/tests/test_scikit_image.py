import unittest
from lime.wrappers.scikit_image import BaseWrapper
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import quickshift
from skimage.data import chelsea
from skimage.util import img_as_float
import numpy as np


class TestBaseWrapper(unittest.TestCase):

    def test_base_wrapper(self):

        obj_with_params = BaseWrapper(a=10, b='message')
        obj_without_params = BaseWrapper()

        def foo_fn():
            return 'bar'

        obj_with_fn = BaseWrapper(foo_fn)
        self.assertEqual(obj_with_params.target_params, {'a': 10, 'b': 'message'})
        self.assertEqual(obj_without_params.target_params, {})
        self.assertEqual(obj_with_fn.target_fn(), 'bar')

    def test__check_params(self):

        def bar_fn(a):
            return str(a)

        class Pipo():

            def __init__(self):
                self.name = 'pipo'

            def __call__(self, message):
                return message

        pipo = Pipo()
        obj_with_valid_fn = BaseWrapper(bar_fn, a=10, b='message')
        obj_with_valid_callable_fn = BaseWrapper(pipo, c=10, d='message')
        obj_with_invalid_fn = BaseWrapper([1, 2, 3], fn_name='invalid')

        # target_fn is not a callable or function/method
        with self.assertRaises(AttributeError):
            obj_with_invalid_fn._check_params('fn_name')

        # parameters is not in target_fn args
        with self.assertRaises(ValueError):
            obj_with_valid_fn._check_params(['c'])
            obj_with_valid_callable_fn._check_params(['e'])

        # params is in target_fn args
        try:
            obj_with_valid_fn._check_params(['a'])
            obj_with_valid_callable_fn._check_params(['message'])
        except Exception:
            self.fail("_check_params() raised an unexpected exception")

        # params is not a dict or list
        with self.assertRaises(TypeError):
            obj_with_valid_fn._check_params(None)
        with self.assertRaises(TypeError):
            obj_with_valid_fn._check_params('param_name')

    def test_set_params(self):

        class Pipo():

            def __init__(self):
                self.name = 'pipo'

            def __call__(self, message):
                return message
        pipo = Pipo()
        obj = BaseWrapper(pipo)

        # argument is set accordingly
        obj.set_params(message='OK')
        self.assertEqual(obj.target_params, {'message': 'OK'})
        self.assertEqual(obj.target_fn(**obj.target_params), 'OK')

        # invalid argument is passed
        try:
            obj = BaseWrapper(Pipo())
            obj.set_params(invalid='KO')
        except Exception:
            self.assertEqual(obj.target_params, {})

    def test_filter_params(self):

        # right arguments are kept and wrong dismmissed
        def baz_fn(a, b, c=True):
            if c:
                return a + b
            else:
                return a
        obj_ = BaseWrapper(baz_fn, a=10, b=100, d=1000)
        self.assertEqual(obj_.filter_params(baz_fn), {'a': 10, 'b': 100})

        # target_params is overriden using 'override' argument
        self.assertEqual(obj_.filter_params(baz_fn, override={'c': False}),
                         {'a': 10, 'b': 100, 'c': False})


class TestSegmentationAlgorithm(unittest.TestCase):

    def test_instanciate_segmentation_algorithm(self):
        img = img_as_float(chelsea()[::2, ::2])

        # wrapped functions provide the same result
        fn = SegmentationAlgorithm('quickshift', kernel_size=3, max_dist=6,
                                   ratio=0.5, random_seed=133)
        fn_result = fn(img)
        original_result = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5,
                                     random_seed=133)

        # same segments
        self.assertTrue(np.array_equal(fn_result, original_result))

    def test_instanciate_slic(self):
        pass

    def test_instanciate_felzenszwalb(self):
        pass


if __name__ == '__main__':
    unittest.main()
