class UnionFind(object):
    def __init__(self, length):
        self._length = length
        self._list = list(i for i in range(length))

    def union(self, idx_a, idx_b):
        set_a = self.find(idx_a)
        set_b = self.find(idx_b)
        if set_a > set_b:
            self._list[set_a] = set_b
        else:
            self._list[set_b] = set_a

    def find(self, idx_a):
        if self._list[idx_a] == idx_a:
            return idx_a
        self._list[idx_a] = self.find(self._list[idx_a])
        return self._list[idx_a]

    def __len__(self):
        return self._length

    def result(self, idx=None):
        if idx is None:
            for i in range(self._length):
                self.find(self._length - 1 - i)
            return self._list
        return self.find(idx)
