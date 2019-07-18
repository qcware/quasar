import unittest
import backend_class_tests

class Test(unittest.TestCase):
    def test_run_unitary(self):
        self.assertTrue(backend_class_tests.run_unitary())
    def test_run_unitary1(self):
        self.assertTrue(backend_class_tests.run_unitary1())
    def test_run_density_matrix(self):
        self.assertTrue(backend_class_tests.run_density_matrix())
    def test_run_density_matrix1(self):
        self.assertTrue(backend_class_tests.run_density_matrix1())
    def test_run_density_matrix_compressed(self):
        self.assertTrue(backend_class_tests.run_density_matrix_compressed())
    def test_run_pauli_expectation_from_statevector(self):
        self.assertTrue(backend_class_tests.run_pauli_expectation_from_statevector())
    def test_run_pauli_expectation_from_measurment(self):
        self.assertTrue(backend_class_tests.run_pauli_expectation_from_measurment())
    def test_bit_reversal_permutation(self):
        self.assertTrue(backend_class_tests.bit_reversal_permutation())
    def test_statevector_bit_reversal_permutation(self):
        self.assertTrue(backend_class_tests.statevector_bit_reversal_permutation())
    def test_statevector_bit_reversal_permutation1(self):
        self.assertTrue(backend_class_tests.statevector_bit_reversal_permutation1())


if __name__ == '__main__':
    unittest.main()
