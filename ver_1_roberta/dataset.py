#coding: utf8

import torch
import torch.utils.data
import torch.nn as nn
import pickle
from ver_1_roberta.param import Param
from ver_1_roberta import *

class _Dataset(torch.utils.data.Dataset):
  def __init__(self, data_files: list):
    data = []
    for f in data_files:
      data.extend(pickle.load(open(f, "rb")))
    self._data = data

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    return self._data[idx]

def _pad_batch_data(batch):
  batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
  subword_ids, masks, intent_ids, tag_ids = list(zip(*batch))
  subword_ids = torch.LongTensor(subword_ids)
  masks = torch.BoolTensor(masks) #(b,l)
  intent_ids = torch.LongTensor(intent_ids)
  tag_ids = torch.LongTensor(tag_ids)
  return subword_ids, masks, intent_ids, tag_ids

def get_batch_data(data_files, epoch_num, shuffle: bool):
  param = Param()
  dataset = _Dataset(data_files)
  data_iter = torch.utils.data.DataLoader(
    dataset, param.batch_size, shuffle=shuffle,
    num_workers=2,
    collate_fn=lambda x: x
  )

  for epoch_id in range(epoch_num):
    for batch in data_iter:
      yield epoch_id, _pad_batch_data(batch)


if __name__ == "__main__":
  param = Param()
  epoch_id, batch = next(iter(get_batch_data(param.train_files, 1, False)))
  subword, mask, intent, tag = batch
  print(subword, mask, intent, tag)