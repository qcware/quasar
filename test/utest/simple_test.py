"""
Some simple tests for unittest 
"""

def test():
    """
    Will always be True
    """
    if 1 + 1 == 2:
        return True
    else:
        return False

def test1(a):
    """
    Returns true if arguement = 1
    """

    if a == 1:
        return True
    else:
        return False

if __name__ == "__main__":
    test()
    test1(1)

