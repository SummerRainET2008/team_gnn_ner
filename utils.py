#coding: utf8

import constants

def get_len(tensor):
  '''
  :param tensor:(b,l)
  :return: (b)
  '''
  mask = tensor.ne(constants.PAD)  # (b,l)
  return mask.sum(dim=-1)

