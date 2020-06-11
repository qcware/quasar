import sortedcontainers

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
        ):

        """ Empty IndexAllocator initializer. """

        self.indices = sortedcontainers.SortedSet()

    @staticmethod
    def build(
        nindex_reserved=0,
        ):

        """ Build a IndexAllocator with a number of pre-reserved indices.

        Params:
            nindex_reserved (int) - number of pre-reserved indices
        Returns:
            (IndexAllocator) - allocator with [0, nindex_reserved) indices
                already allocated.
        """

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
        return self.indices[-1] - self.indices[0] + 1 if len(self.indices) else 0

    @property
    def nindex_sparse(self):
        """ (int) The total number of occupied indices (excluding empty indices). """
        return len(self.indices)

    @property
    def next_index(self):
        """ (int) The next open index that remains unoccupied. """
        if self.nindex == 0: 
            return 0
        elif self.nindex == self.nindex_sparse: 
            return self.max_index + 1
        else: 
            for index in range(self.min_index, self.max_index):
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

class NegativeIndexAllocator(IndexAllocator):

    """ NegativeIndexAllocator functions like IndexAllocator, except that it
        starts allocating at -1 and proceeds in steps of -1.
    """
    
    @property
    def next_index(self):
        """ (int) The next open index that remains unoccupied. """
        if self.nindex == 0: 
            return -1
        elif self.nindex == self.nindex_sparse: 
            return self.min_index - 1
        else: 
            for index in range(self.max_index, self.min_index, -1):
                if index not in self.indices:
                    return index
