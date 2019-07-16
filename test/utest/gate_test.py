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
    def test_set_param(self):
        self.assertTrue(gate_class_tests.set_param())
    def test_set_param1(self):
        self.assertTrue(gate_class_tests.set_param1())
    def test_gateRx(self):
        self.assertTrue(gate_class_tests.gateRx())
    def test_gateRy(self):
        self.assertTrue(gate_class_tests.gateRy())
    def test_gateRz(self):
        self.assertTrue(gate_class_tests.gateRz())
    def test_gateSO4(self):
        self.assertTrue(gate_class_tests.gateSO4())
    def test_gateSO42(self):
        self.assertTrue(gate_class_tests.gateSO42())
    def test_gateCF(self):
        self.assertTrue(gate_class_tests.gateCF())
    def test_gateU1(self):
        self.assertTrue(gate_class_tests.gateU1())
    def test_gateU2(self):
        self.assertTrue(gate_class_tests.gateU2())
    def test_init_controlled_gate(self):
        self.assertTrue(gate_class_tests.init_controlled_gate())

if __name__ == '__main__':
    unittest.main()
