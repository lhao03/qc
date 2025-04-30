import numpy as np
import scipy as sp
import h5py


def dump_sparse2_h5(file_str, sparse_array):
  '''
  Dump scipy sparse array to a .h5 file.

  Args:
      file_str (str): File name.
      sparse_array (scipy.sparse.csc_matrix): Sparse array to be dumped.

  Returns:
      None
  '''
  with h5py.File(file_str, "w") as f:
      # Save data, indices, and indptr sparse_arrays with gzip compression
      f.create_dataset("data", data=sparse_array.data, compression="gzip")
      f.create_dataset("indices", data=sparse_array.indices, compression="gzip")
      f.create_dataset("indptr", data=sparse_array.indptr, compression="gzip")
      f.attrs["shape"] = sparse_array.shape  # Store shape as an attribute







def load_h5_sparse (file_str):
  '''
  Load scipy sparse stored as a .h5 file.

  Args:
      file_str (str): File name.

  Returns:
      scipy.sparse.csc_matrix: Sparse array loaded from h5 file.
  '''
  with h5py.File(file_str, "r") as f:
    # Load data, indices, and indptr
    data = f["data"][:]
    indices = f["indices"][:]
    indptr = f["indptr"][:]
    shape = f.attrs["shape"]
    return sp.sparse.csc_matrix((data, indices, indptr), shape=shape)








def dump_list_of_sparse2_h5(file_str, array_list):
  '''
  Dump a list of scipy sparse arrays as a .h5 file

  Args:
      file_str (str): File name.
      array_list (list): List of scipy.sparse.csc_matrix objects to be dumped.

  Returns:
      None
  '''
  with h5py.File(file_str, "w") as f:
    for i, sparse_array in enumerate(array_list):
      group = f.create_group(f"array_{i}")

      # Store CSC components: data, indices, indptr, and shape
      group.create_dataset('data', data=sparse_array.data, compression='gzip')
      group.create_dataset('indices', data=sparse_array.indices, compression='gzip')
      group.create_dataset('indptr', data=sparse_array.indptr, compression='gzip')
      group.attrs['shape'] = sparse_array.shape








def load_h5_list_of_sparse (file_str, index = None):    #When index is provided, the only one sparse array at the index in the list is returned.
  '''
  Load a list of scipy sparse arrays stored as a .h5 file.

  Args:
      file_str (str): File name.
      index (int, optional): If provided, only the sparse array at that index will be loaded. Defaults to None.

  Returns:
      if index == None --> list: List of scipy.sparse.csc_matrix objects
                  else --> scipy.sparse.csc_matrix: Sparse array at index = index
  '''
  if index == None:
    with h5py.File(file_str, "r") as f:
      array_list = []
      for i in range(len(f.keys())):
        group = f[f"array_{i}"]
        # Load data, indices, and indptr
        data = group['data'][:]
        indices = group['indices'][:]
        indptr = group['indptr'][:]
        shape = group.attrs['shape']
        array_list.append(sp.sparse.csc_matrix((data, indices, indptr), shape=shape))
      return array_list
  else:
    with h5py.File(file_str, 'r') as f:
      group = f[f"array_{index}"]

      # Load CSC components
      data = group['data'][:]
      indices = group['indices'][:]
      indptr = group['indptr'][:]
      shape = group.attrs['shape']

      # Reconstruct the sparse array in CSC format
      return (sp.sparse.csc_matrix((data, indices, indptr), shape=shape))









def dump_ndarray_h5(file_str, array, label = None):
  '''
  Save a numpy dense array in .h5 file.

  Args:
      file_str (str): File name.
      array (np.array): Array to be saved.
      label (str, optional): Label for the array. Defaults to None.

  Returns:
      None
  '''
  if label == None:
    label = "ndarray"
  with h5py.File(file_str, "w") as f:
    f.create_dataset(label, data=array, compression="gzip")







def load_h5_ndarray (file_str, label = None):
  '''
  Load a numpy dense array from .h5 file.

  Args:
      file_str (str): File name.
      label (str, optional): Label for the array. Defaults to None.

  Returns:
      np.array: Array loaded from h5 file.
  '''
  if label == None:
    label = "ndarray"
  with h5py.File(file_str, "r") as f:
    return f[label][:]