import unittest
import simple_test
import some_tests

class Test(unittest.TestCase):
    def test_test(self):
        # should always be true
        self.assertTrue(simple_test.test())
    def test_test1(self, one=1):
        # First statement should return True. Second should return False.
        self.assertTrue(simple_test.test1(1))
        # self.assertTrue(simple_test.test1(2))
    
    def test_init_circuit(self):
        self.assertTrue(some_tests.init_circuit())
        
    
if __name__ == '__main__':
    unittest.main()