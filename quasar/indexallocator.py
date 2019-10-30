import sortedcontainers
from operator import neg

class IndexAllocator(object):

    """ IndexAllocator provides a utility object to track index
        allocation/deallocation, e.g., for reuse of ancilla indices.

        Example use:

        >>> allocator = IndexAllocator()
        >>> index0 = allocator.allocate()
        >>> index1 = allocator.allocate()
        >>> index2 = allocator.allocate()
        >>> print(allocator.next_index)
        >>> allocator.deallocate(index1)
        >>> print(allocator.next_index)
        
        For convenience, one can initialize a IndexAllocator with a number of
        pre-allocated indices via the build method:

        >>> allocator = IndexAllocator.build(10)
        >>> print(allocator.next_index)

        Negative indices are permitted throughout - next_index is always
        greater than min_index:

        >>> allocator = IndexAllocator()
        >>> allocator.allocate(-5)
        >>> print(allocator.next_index)

        Attributes:
            indices (SortedSet of int) - currently allocated indices.
    """
    def __init__(
        self,
        negative_convention=False
        ):

        """ Empty IndexAllocator initializer. """

        self.negative_convention = negative_convention

        if negative_convention:
            self.indices = sortedcontainers.SortedSet(key=neg)
        else:
            self.indices = sortedcontainers.SortedSet()

    @staticmethod
    def build(
        nindex_reserved=0,
        negative_convention = False
        ):

        """ Build a IndexAllocator with a number of pre-reserved indices.

        Params:
            nindex_reserved (int) - number of pre-reserved indices
        Returns:
            (IndexAllocator) - allocator with [0, nindex_reserved) indices
                already allocated.
        """

        if negative_convention:
            allocator = IndexAllocator(negative_convention)
            for index in range(-1, -nindex_reserved - 1, -1):
                allocator.allocate(index)
            return allocator

        else:
            allocator = IndexAllocator()
            for index in range(nindex_reserved):
                allocator.allocate()
            return allocator

    @property
    def min_index(self):
        """ (int) The minimum occupied index (or 0 if no occupied indices) """
        return self.indices[0] if len(self.indices) else 0
    
    @property
    def max_index(self):
        """ (int) The maximum occupied index (or -1 if no occupied indices) """
        return self.indices[-1] if len(self.indices) else -1

    @property
    def nindex(self):
        """ (int) The total number of indices (including empty indices). """

        if self.negative_convention:
            return abs(self.indices[-1]) - abs(self.indices[0]) + 1 if len(self.indices) else 0

        else:
            return self.indices[-1] - self.indices[0] + 1 if len(self.indices) else 0

    @property
    def nindex_sparse(self):
        """ (int) The total number of occupied indices (excluding empty indices). """
        return len(self.indices)

    @property
    def next_index(self):
        """ (int) The next open index that remains unoccupied. """
        if self.nindex == 0: 
            if self.negative_convention:
                return - 1
            else:
                return 0

        elif self.nindex == self.nindex_sparse: 
            if self.negative_convention:
                return self.max_index - 1
            else:
                return self.max_index + 1
        else: 
            if self.negative_convention:
                for index in range(self.min_index, self.max_index + 1, -1):
                    if index not in self.indices:
                        return index
            else:
                for index in range(self.min_index, self.max_index + 1):
                    if index not in self.indices:
                        return index

    def allocate(self, index=None):
        """ Allocate an index.

        Params:
            index (int or None) - index to allocate. If None, allocates
                self.next_index. If index is already allocated, a RuntimeError
                will be raised.
        Return:
            (int) - allocated index, useful in case input index is None
                and self.next_index is allocated.
        """

        if index != None and index >= 0 and self.negative_convention: 
            raise RuntimeError('No non-negative indices may be allocated when following negative convention.')

        if index is None: index = self.next_index
        if index in self.indices: raise RuntimeError('Already allocated: %d' % index)
        self.indices.add(index)
        return index

    def deallocate(self, index):
        """ Deallocate an index.
        
        Params:
            index (int) - index to deallocate. If index is not allocated,
                a RuntimeError will be raised. 
        """
        
        if index not in self.indices: raise RuntimeError('Not allocated: %d' % index)
        self.indices.remove(index)
