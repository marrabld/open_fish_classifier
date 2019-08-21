from multiprocessing import Pool
import time


def stupid_parallel(function, nprocesses=None):
    """
    Stolen from here https://valentinoetal.wordpress.com/2014/06/10/stupid-parallel-pseudo-decorator-in-python/

    Works similar to a decorator to paralelize "stupidly parallel"
    problems. Decorators and multiprocessing don't play nicely because
    of naming issues.

    Inputs
    
    function : the function that will be parallelized. The FIRST
        argument is the one to be iterated on (in parallel). The other
        arguments are the same in all the parallel runs of the function
        (they can be named or unnamedarguments).
    nprocesses : int, the number of processes to run. Default is None.
        It is passed to multiprocessing.Pool (see that for details).

    Output
    
    A paralelized function. DO NOT NAME IT THE SAME AS THE INPUT
    FUNCTION.

    Example
    
   
    def _square_and_offset(value, offset=0):
        return value**2 + offset

    parallel_square_and_offset = stupid_parallel(_square_and_offset,
                                                 nprocesses=5)
    print square_and_offset_parallel(range(10), offset=3)
    > [3, 4, 7, 12, 19, 28, 39, 52, 67, 84]

    """

    def apply(iterable_values, *args, **kwargs):
        args = list(args)
        try:
            p = Pool(nprocesses)
            result = [p.apply_async(function, args=[value] + args,
                                    kwds=kwargs)
                      for value in iterable_values]
        finally:
            p.close()
            p.join()
        return [r.get() for r in result]

    return apply


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te - ts))
        return result

    return timed
