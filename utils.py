#coding: utf8

import Constants

def get_len(tensor):
  '''
  :param tensor:(b,l)
  :return: (b)
  '''
  mask = tensor.ne(Constants.PAD)  # (b,l)
  return mask.sum(dim=-1)

