from libc.stdlib cimport malloc, free, realloc


__all__ = ['BoundingBox', 'ngrams', 'unigrams', 'bigrams', 'trigrams']

cdef list _ngram(list words, int n=2):
    cdef int i;
    return list(zip(*[words[i:] for i in range(n)]))

cpdef list ngrams(list words, int start=1, int stop=10):
    cdef int i
    return [_ngram(words, i) for i in range(start,stop)]
    
cpdef list unigrams(list words):
    return ngrams(words, stop=2)

cpdef list bigrams(list words):
    return ngrams(words, start=2, stop=3)

cpdef list trigrams(list words):
    return ngrams(words, start=3, stop=4)


cdef class BoundingBox:
    cdef float x0, x1, y0, y1
    cpdef public tuple coords
    
    def __init__(self, float x0, float y0, float x1, float y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.coords = (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_xywh(cls, float x, float y, float w, float h):
        return cls(x, y, x+w, y+h)

    cpdef bint intersects(self, BoundingBox other):
        return (((other.x0 <= self.x0 <= other.x1) |
                 (other.x0 <= self.x1 <= other.x1)) &
                ((other.y0 <= self.y0 <= other.y1) | 
                 (other.y0 <= self.y1 <= other.y1)))

