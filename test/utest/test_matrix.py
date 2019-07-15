import quasar
import numpy as np

"""
Test "Matrix" Class
"""

def one_qubit_constant_matrices():
    """
    Validate the shape and dtype of 1 qubit matrix in "Matrix" class.
    """
    matrix_list = ['I', 'X', 'Y', 'Z', 'S', 'T', 'H', 'Rx2', 'Rx2T']
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)
        if not m.shape==(2,2):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True
    
    
def two_qubit_constant_matrices():
    """
    Validate the shape and dtype of 2 qubit matrix in "Matrix" class.
    """
    matrix_list = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ', 'CX', 'CY', 'CZ', 'CS', 'SWAP']
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)
        if not m.shape==(4,4):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True


def three_qubit_constant_matrices():
    """
    Validate the shape and dtype of 3 qubit matrix in "Matrix" class.
    """
    matrix_list = ['CCX', 'CSWAP']
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)
        if not m.shape==(8,8):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True


def one_qubit_1param_matrices():
    """
    Validate the shape and dtype of 1 qubit matrix in "Matrix" class.
    """
    matrix_list = ['Rx', 'Ry', 'Rz', 'u1', 'Rz_ion']
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)(np.random.rand())
        if not m.shape==(2,2):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True    
    
    
def one_qubit_2param_matrices():
    """
    Validate the shape and dtype of 1 qubit matrix in "Matrix" class.
    """
    matrix_list = ['u2','R_ion',]
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)(*np.random.rand(2))
        if not m.shape==(2,2):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True     
    

def one_qubit_3param_matrices():
    """
    Validate the shape and dtype of 1 qubit matrix in "Matrix" class.
    """
    matrix_list = ['u3']
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)(*np.random.rand(3))
        if not m.shape==(2,2):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True 


def two_qubit_1param_matrices():
    """
    Validate the shape and dtype of 1 qubit matrix in "Matrix" class.
    """
    matrix_list = ['XX_ion']
    for m_name in matrix_list:
        m = getattr(quasar.Matrix, m_name)(np.random.rand())
        if not m.shape==(4,4):
            return False
        if not m.dtype == np.complex128:
            return False
            
    return True     
    
    
# print(two_qubit_1param_matrices())
    
        
        














