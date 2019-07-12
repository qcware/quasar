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
    def test_ntime(self):
        self.assertTrue(some_tests.ntime())
    def test_ngate(self):
        self.assertTrue(some_tests.ngate())
    def test_ngate1(self):
        self.assertTrue(some_tests.ngate1())
    def test_ngate2(self):
        self.assertTrue(some_tests.ngate2())
    def test_add_gate(self):
        self.assertTrue(some_tests.add_gate())
    def test_gate(self):
        self.assertTrue(some_tests.gate())
    def test_copy(self):
        self.assertTrue(some_tests.copy())
    def test_subset(self):
        self.assertTrue(some_tests.subset())
    def test_concatenate(self):
        self.assertTrue(some_tests.concatenate())
    def test_deadjoin(self):
        self.assertTrue(some_tests.deadjoin())
    def test_adjoin(self):
        self.assertTrue(some_tests.adjoin())
    def test_reversed(self):
        self.assertTrue(some_tests.test_reversed())

if __name__ == '__main__':
    unittest.main()

