import unittest
import simple_test
import test_circuit


class Test(unittest.TestCase):
    def test_test(self):
        # should always be true
        self.assertTrue(simple_test.test())
    def test_test1(self, one=1):
        # First statement should return True. Second should return False.
        self.assertTrue(simple_test.test1(1))
        # self.assertTrue(simple_test.test1(2))
    
    def test_circuit_class(self):
        # test: Circult.simulate()
        self.assertTrue(test_circuit.simulate())
        self.assertTrue(test_circuit.simulate_steps())
        self.assertTrue(test_circuit.apply_gate_1())
        self.assertTrue(test_circuit.apply_gate_1_format())
        self.assertTrue(test_circuit.apply_gate_2())
        self.assertTrue(test_circuit.apply_gate_2_format())
        self.assertTrue(test_circuit.apply_gate_3())
        self.assertTrue(test_circuit.apply_gate_3_format())
        
    
if __name__ == '__main__':
    unittest.main()
