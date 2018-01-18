import traitlets as tl
import numpy as np

#def cached_property(dependencies):
def cached_property(func):
    @property
    def f(self):
        cache_name = '_cached_' + func.__name__
        if hasattr(self, cache_name):
            cache_val = getattr(self, cache_name)
        else: cache_val = None
        if cache_val is not None:
            return cache_val
        cache_val = func(self)
        setattr(self, cache_name, cache_val)
        return cache_val
    return f
    
def clear_cache(self, change, attrs):
    if (change['old'] is None and change['new'] is not None) or \
               np.any(np.array(change['old']) != np.array(change['new'])):
        for attr in attrs:
            setattr(self, '_cached_' + attr, None)

    
if __name__ == "__main__":
    class Dum(tl.HasTraits):
        @cached_property
        def test(self):
            print ("Calculating Test")
            return 'test_prints' + str(self.lala)
        
        lala = tl.Int(0)
        
        @tl.observe('lala')
        def lalaobs(self, change):
            clear_cache(self, change, ['test'])

    d = Dum()
    print (d.test, d.test)
    d.lala = 10
    print (d.test, d.test)
