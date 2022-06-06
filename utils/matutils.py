import numpy as np

def process_mat(mat):
  new_mat = np.nan_to_num(mat)
  mask = new_mat != 0

  max_val = np.max(new_mat)
  min_val = np.min(new_mat[mask])
  new_mat[~mask]= min_val
  
  
  new_mat = np.log(new_mat)
  range_log = np.log(max_val)-np.log(min_val)
  new_mat -= np.log(min_val)
  new_mat /= range_log
  return new_mat