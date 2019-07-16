import unittest
import gate_class_tests

class Test(unittest.TestCase):
    def test_init_gate(self):
        self.assertTrue(gate_class_tests.init_gate())
    def test_init_param_gate(self):
        self.assertTrue(gate_class_tests.init_param_gate())
    def test_same_unitary1(self):
        self.assertTrue(gate_class_tests.same_unitary1())
    def test_same_unitary2(self):
        self.assertTrue(gate_class_tests.same_unitary2())
    def test_same_unitary3(self):
        self.assertTrue(gate_class_tests.same_unitary3())
    def test_U(self):
        self.assertTrue(gate_class_tests.U())
    def test_U1(self):
        self.assertTrue(gate_class_tests.U1())
    def test_copy(self):
        self.assertTrue(gate_class_tests.copy())
    def test_set_params(self):
        self.assertTrue(gate_class_tests.set_params())
    def test_set_params1(self):
        self.assertTrue(gate_class_tests.set_params1())

if __name__ == '__main__':
    unittest.main()
