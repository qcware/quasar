import quasar
import numpy as np

"""
Test "MeasurementResult" Class
"""

def util_get_measurement():
    """
    A utility function for the testing functions below.
    """
    ket1 = quasar.Ket.from_int(9,5)
    ket2 = quasar.Ket.from_int(19,5)
    ket3 = quasar.Ket.from_int(29,5)
    m_result = quasar.MeasurementResult({ket1:101, ket2:202, ket3:303})
    return m_result


def initialization():
    """
    Validate the initialization (__init__()) of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()

    return [x for x in m_result.items()]==[('01001', 101), ('10011', 202), ('11101', 303)]

    
def contains():
    """
    Validate the __contains__() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()
    if not '01001' in m_result:
        return False
    
    return True
    
    
def getitem():
    """
    Validate the __getitem__() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()

    return m_result['10011']==202
    

def setitem():
    """
    Validate the __setitem__() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()
    m_result['11111']=505

    return m_result['11111']==505    
    

def get():
    """
    Validate the get() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()

    return m_result.get('10011')==202    
    
    
def setdefault():
    """
    Validate the get() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()
    m_result.setdefault('10011', 8)
    m_result.setdefault('11111', 8)

    return m_result.get('10011')==202 and m_result.get('11111')==8   
    
    
def N():
    """
    Validate the N() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()

    return m_result.N==5
    
    
def nmeasurement():
    """
    Validate the nmeasurement() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()

    return m_result.nmeasurement==606    
    
    
def str():
    """
    Validate the __str__() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()

    return m_result.__str__().replace('\n', '')=='|01001> : 101|10011> : 202|11101> : 303'      
    
    
def subset():
    """
    Validate the subset() of the "MeasurementResult" class.
    """
    m_result = util_get_measurement()
    m_result_subset = m_result.subset([0,3])

    return [x for x in m_result_subset.items()]==[('00', 101), ('11', 202), ('10', 303)]
    
    
    
