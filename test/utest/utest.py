import unittest
import simple_test

class Test(unittest.TestCase):
    def test_test(self):
        # should always be true
        self.assertTrue(simple_test.test())
    def test_test1(self, one=1):
        # First statement should return True. Second should return False.
        self.assertTrue(simple_test.test1(1))
        self.assertTrue(simple_test.test1(2))
    
if __name__ == '__main__':
    unittest.main()

