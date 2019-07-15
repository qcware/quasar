import quasar
import numpy as np

"""
Test "Ket" Class
"""

def initialization():
    """
    Validate the initialization of the "Ket" class.
    """
    ket_str = '{0:05b}'.format(9)
    ket = quasar.Ket(ket_str)
            
    return ket=='01001'
    
    
def getitem():
    """
    Validate the "Ket.__getitem__" .
    """
    ket_str = '{0:05b}'.format(9)
    ket = quasar.Ket(ket_str)
            
    return ket.__getitem__(4)==1

    
def N():
    """
    Validate the "Ket.N" .
    """
    ket_str = '{0:05b}'.format(9)
    ket = quasar.Ket(ket_str)
            
    return ket.N==5    
    
    
def from_int():
    """
    Validate the "Ket.from_int" .
    """
    ket = quasar.Ket.from_int(9,5)
    
    return ket=='01001'
    
    
    
    






