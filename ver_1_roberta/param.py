#coding: utf8

import os
from ver_1_roberta import *
from pa_nlp.pytorch.estimator.param import ParamBase

class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("ver_1_roberta")

    self.train_files = [
      f"{self.path_feat}/train.pydict"
    ]
    self.vali_file = f"{self.path_feat}/vali.pydict"
    self.test_files = []

    self.pretrained_model = os.path.expanduser(
      "~/pretrained_models/roberta/roberta.large/",
    )
    self.file_for_label_tokenizer = 'data/snips/train/intent_seq.out'

    self.optimizer_name = "Adam"
    self.bert_lr = 1.5e-5
    self.lr = 5e-2
    self.epoch_num   = 50
    self.batch_size_one_gpu  = 32
    self.dropout = 0.5
    self.max_seq_len = 50
    self.d_word = 1024
    self.d_hidden = 150 # must be divisible by the number of attention heads
    self.src_vocab_size = 52000
    self.tgt_vocab_size = TAGS
    self.n_intent = INTENTS
    self.train_sample_num = 13084
    self.eval_gap_instance_num = 1280

    self.use_polynormial_decay = False
    self.incremental_train = False