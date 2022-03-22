
import numpy as np
import numpy.lib.mixins 
from numbers import Number
import sympy

class PartialMatrix(numpy.lib.mixins.NDArrayOperatorsMixin): 
  @classmethod 
  def from_entries(cls,indices,entries,shape=None):
    indices = np.array(indices) 
    entries = np.array(entries)
    #If shape is not provided, take it to be smallest 
    #possible array containing all given entries
    if(shape is None):
      shape = tuple(np.amax(indices,0)+1)
    obj = cls.__new__(cls)
    super(PartialMatrix, obj).__init__()  
    obj.indices = indices
    obj.entries = entries
    obj.shape = shape
    return obj
  def __init__(self, X, mask = None): 
    X = np.array(X) 
    mask = np.ones(X.shape) if mask is None else np.array(mask)
    self.indices = np.argwhere(np.array(mask) == 1)
    self.entries = X[mask==1] 
    self.shape = X.shape 
  def zero_filled(self): 
    known = np.zeros(self.shape) 
    known[tuple(self.indices.T)] = self.entries 
    return known 
  def get_mask(self): 
    mask = np.zeros(self.shape,dtype = "int64") 
    mask[tuple(self.indices.T)] = 1 
    return mask 
  def get_mask_repr(self): 
    mask = self.get_mask() 
    known = np.zeros(self.shape) 
    known[mask==1] = self.entries 
    return known,mask
  def __getitem__(self,index):
    known_index = (self.indices==index).all(1)
    if not known_index.any():
      return None
    else:
      return self.entries[known_index].item()
  def __setitem__(self,index,newvalue):
    known_index = (self.indices==index).all(1)
    if not known_index.any():
      self.entries = np.append(self.entries,newvalue)
      self.indices = np.vstack([self.indices,index])
    else:
      self.entries[known_index] = newvalue
  def __repr__(self): 
    known,mask = self.get_mask_repr() 
    strknown = known.astype("str") 
    maxlen = np.max(np.char.str_len(strknown)) 
    strknown[mask == 0] = "__"
    return np.array2string(strknown, 
                           separator = ",", 
                           formatter={'str_kind': lambda x: x}) 
  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    scalars = []
    shape = None
    for i in inputs:
      if isinstance(i,Number):
        scalars.append(i)
      elif isinstance(i,self.__class__):
        if shape is not None:
          if not np.array_equal(shape,i.shape):
            raise TypeError("Inconsistent shapes")
        else:
          shape = i.shape
        scalars.append(i.entries)
    return PartialMatrix.from_entries(
                          self.indices,
                          ufunc(*scalars,**kwargs),
                          shape = shape)
  def num_unknowns(self):
    return np.sum(mask!=1)
  def sympy_repr(self,symmetric = False):
    #not that fast, probably could be better
    known,mask = self.get_mask_repr()
    unknown_idx = np.argwhere(mask!=1)
    if symmetric:
      unknown_idx = unknown_idx[unknown_idx[:,0]<unknown_idx[:,1],:]
    out = sympy.Matrix(known)
    x = sympy.symbols(['x'+str(i) for i in range(unknown_idx.shape[0])])
    for i,symb in enumerate(x):
      out[tuple(unknown_idx[i,:])] = symb
      if symmetric:
        out[tuple(np.flip(unknown_idx[i,:]))] = symb
    return out

  def det_func(self,*args):
    mat = self.sympy_repr()
    func = sympy.lambdify(list(mat.free_symbols),
                          mat.det(),modules = "numpy")
    if len(args)==0:
      return func
    else:
      return func(*args)
  def det_poly(self,
               dict_repr = True,
               symmetric = False):
    mat = self.sympy_repr(symmetric = symmetric)
    det = sympy.poly(mat.det(),
                        gens = list(mat.free_symbols))
    if not dict_repr:
      return det
    else:
      return det.as_dict()
  def minor_poly(self,
                 idx,
                 dict_repr = True,
                 symmetric = False):
    mat = self.sympy_repr(symmetric = symmetric)
    minor = sympy.poly(mat.minor(*idx),
                        gens = list(mat.free_symbols))
    if not dict_repr:
      return minor
    else:
      return minor.as_dict()
  def p_minor_polys(self,
                    dict_repr = True,
                    symmetric = False):
    minors = [self.minor_poly((i,i),dict_repr,symmetric)
              for i in range(self.shape[0])]
    out = minors
    if dict_repr:
      allkeys = set()
      for i,minor in enumerate(minors):
        allkeys = allkeys.union(minor.keys())
      out = dict(zip(allkeys,
                     [()]*len(allkeys)))
      for i,minor in enumerate(minors):
        for j,key in enumerate(allkeys):
          if key in minor.keys():
            out[key] = out[key] + (minor[key],)
          else:
            out[key] = out[key] + (0,)
    return out
if __name__ == "__main__":
  x = [[13,3,2],[2,3,1],[1,1,1]] 
  x = np.random.randn(10,10)
  mask = np.ones((10,10))
  mask[3,2] = 0
  mask[2,3] = 0
  p = PartialMatrix(x,mask)
  print(p.sympy_repr(symmetric=True))
  print(p.num_unknowns())
  print(p.p_minor_polys(dict_repr = False,symmetric = True))



