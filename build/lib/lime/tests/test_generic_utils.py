import unittest
import sys
from lime.utils.generic_utils import has_arg


class TestGenericUtils(unittest.TestCase):

    def test_has_arg(self):
        # fn is callable / is not callable

        class FooNotCallable:

            def __init__(self, word):
                self.message = word

        class FooCallable:

            def __init__(self, word):
                self.message = word

            def __call__(self, message):
                return message

            def positional_argument_call(self, arg1):
                return self.message

            def multiple_positional_arguments_call(self, *args):
                res = []
                for a in args:
                    res.append(a)
                return res

            def keyword_argument_call(self, filter_=True):
                res = self.message
                if filter_:
                    res = 'KO'
                return res

            def multiple_keyword_arguments_call(self, arg1='1', arg2='2'):
                return self.message + arg1 + arg2

            def undefined_keyword_arguments_call(self, **kwargs):
                res = self.message
                for a in kwargs:
                    res = res + a
                return a

        foo_callable = FooCallable('OK')
        self.assertTrue(has_arg(foo_callable, 'message'))

        if sys.version_info < (3,):
            foo_not_callable = FooNotCallable('KO')
            self.assertFalse(has_arg(foo_not_callable, 'message'))
        elif sys.version_info < (3, 6):
            with self.assertRaises(TypeError):
                foo_not_callable = FooNotCallable('KO')
                has_arg(foo_not_callable, 'message')

        # Python 2, argument in / not in valid arguments / keyword arguments
        if sys.version_info < (3,):
            self.assertFalse(has_arg(foo_callable, 'invalid_arg'))
            self.assertTrue(has_arg(foo_callable.positional_argument_call, 'arg1'))
            self.assertFalse(has_arg(foo_callable.multiple_positional_arguments_call, 'argX'))
            self.assertFalse(has_arg(foo_callable.keyword_argument_call, 'argX'))
            self.assertTrue(has_arg(foo_callable.keyword_argument_call, 'filter_'))
            self.assertTrue(has_arg(foo_callable.multiple_keyword_arguments_call, 'arg2'))
            self.assertFalse(has_arg(foo_callable.multiple_keyword_arguments_call, 'arg3'))
            self.assertFalse(has_arg(foo_callable.undefined_keyword_arguments_call, 'argX'))
        # Python 3, argument in / not in valid arguments / keyword arguments
        elif sys.version_info < (3, 6):
            self.assertFalse(has_arg(foo_callable, 'invalid_arg'))
            self.assertTrue(has_arg(foo_callable.positional_argument_call, 'arg1'))
            self.assertFalse(has_arg(foo_callable.multiple_positional_arguments_call, 'argX'))
            self.assertFalse(has_arg(foo_callable.keyword_argument_call, 'argX'))
            self.assertTrue(has_arg(foo_callable.keyword_argument_call, 'filter_'))
            self.assertTrue(has_arg(foo_callable.multiple_keyword_arguments_call, 'arg2'))
            self.assertFalse(has_arg(foo_callable.multiple_keyword_arguments_call, 'arg3'))
            self.assertFalse(has_arg(foo_callable.undefined_keyword_arguments_call, 'argX'))
        else:
            self.assertFalse(has_arg(foo_callable, 'invalid_arg'))
            self.assertTrue(has_arg(foo_callable.positional_argument_call, 'arg1'))
            self.assertFalse(has_arg(foo_callable.multiple_positional_arguments_call, 'argX'))
            self.assertFalse(has_arg(foo_callable.keyword_argument_call, 'argX'))
            self.assertTrue(has_arg(foo_callable.keyword_argument_call, 'filter_'))
            self.assertTrue(has_arg(foo_callable.multiple_keyword_arguments_call, 'arg2'))
            self.assertFalse(has_arg(foo_callable.multiple_keyword_arguments_call, 'arg3'))
            self.assertFalse(has_arg(foo_callable.undefined_keyword_arguments_call, 'argX'))
        # argname is None
        self.assertFalse(has_arg(foo_callable, None))


if __name__ == '__main__':
    unittest.main()
