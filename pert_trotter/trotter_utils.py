import numpy as np
import scipy as sp
import gc






def get_trotter_gens(frags_list = list, ord = int):
  '''
  Get the sequence of Hamiltonian fragments that generate unitaries of a Trotter product formula of order "ord". 
  For orders greater than 1, the formula is based on the Suzuki-Trotter decomposition.
  Note, here order "n" leads to a product formula of leading order error n+1 in time step.

  Args:
      frags_list (list): List of Hamiltonian fragments.
      ord (int): Order of the Trotter product formula.
  
  Returns:
      list: List of Trotter generators.
  '''

  if ord == 1:
    return frags_list
  else:
    S2 = [A*0.5 for A in frags_list] + [A*0.5 for A in frags_list[::-1]]
    for p in range(4, ord+1, 2):
      sp = (4-4**(1/(p-1)))**(-1)
      S2a = [A*sp for A in S2]
      S2b = [A*(1-4*sp) for A in S2]
      S2 = S2a + S2a + S2b + S2a + S2a
      del S2a, S2b
      gc.collect()
    
    return S2





def get_suzuki_gens(frags_list = list, ord = int):
  '''
  Get the sequence of Hamiltonian fragments that generate unitaries of a Suzuki product formula of order "ord".
  Note, here order "n" leads to a product formula of leading order error 2n+1 in time step.
  So get_suzuki_gens(frags_list, 1) = get_trotter_gens(frags_list, 2), and
  get_suzuki_gens(frags_list, 2) = get_trotter_gens(frags_list, 4).

  Args:
      frags_list (list): List of Hamiltonian fragments.
      ord (int): Order of the Suzuki product formula.
  
  Returns:
      list: List of Suzuki generators.
  '''
  return get_trotter_gens(frags_list, 2*ord)