import quasar
import unittest
import numpy as np

class TestMatrix(unittest.TestCase):

    def helper_test_matrices(
        self,
        quasar_matrices,
        reference_matrices,
        ):

        for key, reference_matrix in reference_matrices.items():
            quasar_matrix = quasar_matrices[key]
            self.assertIsInstance(quasar_matrix, np.ndarray)
            self.assertEqual(quasar_matrix.dtype, np.complex128)
            self.assertEqual(quasar_matrix.shape, reference_matrix.shape)
            self.assertTrue(np.max(np.abs(quasar_matrix - reference_matrix)) < 1.0E-14)

    def test_one_qubit_raw(self):

        quasar_matrices = {
            'I'  : quasar.Matrix.I,
            'X'  : quasar.Matrix.X,
            'Y'  : quasar.Matrix.Y,
            'Z'  : quasar.Matrix.Z,
            'S'  : quasar.Matrix.S,
            'ST' : quasar.Matrix.ST,
            'T'  : quasar.Matrix.T,
            'TT' : quasar.Matrix.TT,
            'H'  : quasar.Matrix.H,
        }

        reference_matrices = {
            'I'  : np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
            'X'  : np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            'Y'  : np.array([[0.0, -1.0j], [+1.0j, 0.0]], dtype=np.complex128),
            'Z'  : np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
            'S'  : np.array([[1.0, 0.0], [0.0, +1.0j]], dtype=np.complex128),
            'ST' : np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128),
            'T'  : np.array([[1.0, 0.0], [0.0, np.cos(np.pi / 4.0) + 1.j * np.sin(np.pi / 4.0)]], dtype=np.complex128),
            'TT' : np.array([[1.0, 0.0], [0.0, np.cos(np.pi / 4.0) - 1.j * np.sin(np.pi / 4.0)]], dtype=np.complex128),
            'H'  : np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
        }

        self.helper_test_matrices(quasar_matrices, reference_matrices)

            
        
 
