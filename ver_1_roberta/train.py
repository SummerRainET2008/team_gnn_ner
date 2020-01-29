#coding: utf8

import torch
import torch.nn as nn
import torch.utils.data
import optparse
import time

from pa_nlp import nlp
from pa_nlp.nlp import Logger
from pa_nlp.pytorch.estimator.train import TrainerBase
from ver_1_roberta import *
from ver_1_roberta.dataset import get_batch_data
from ver_1_roberta._model import SLSTM
from ver_1_roberta.evaluate import compute_f1_score, get_sent_acc
from ver_1_roberta.make_features import LabelTokenizer
from ver_1_roberta.param import Param


class Trainer(TrainerBase):
  def __init__(self):
    param = Param()
    param.verify()

    self._opt_vali_error = -100
    model = SLSTM(
      d_word=param.d_word, d_hidden=param.d_hidden,
      n_src_vocab=param.src_vocab_size, n_tgt_vocab=param.tgt_vocab_size,
      n_intent=param.n_intent, dropout=param.dropout
    )

    optimizer_parameters = [
      {
        'params': [
          p for n, p in model.named_parameters() if "roberta" in n
        ],
        'lr': param.bert_lr,
      },
      {
        'params': [
          p for n, p in model.named_parameters() if "roberta" not in n
        ],
        'lr': param.lr,
      },
    ]

    optimizer = getattr(torch.optim, param.optimizer_name)(
      optimizer_parameters
    )

    super(Trainer, self).__init__(
      param, model,
      get_batch_data(param.train_files, param.epoch_num, True),
      optimizer
    )
  def _train_one_batch(self, b_subword_ids, b_masks, b_intent_ids, b_tag_ids):
    '''
    Args:
      b_subword_ids: (b,l,d)
      b_masks: (b,l)
      b_intent_ids: (b)
      b_tag_ids: (b,l)
    Returns:

    '''
    slot_logit, intent_logit = self.predict(b_subword_ids, b_masks)
    slot_logit = slot_logit.view(-1, self._param.tgt_vocab_size) # (l*b,n_tgt)
    intent_logit = intent_logit.view(-1, self._param.n_intent)

    slot_gold = b_tag_ids.transpose(0, 1).contiguous().view(-1) # (b*l)
    intent_gold = b_intent_ids.view(-1) # (b)
    slot_loss = nn.functional.cross_entropy(
      slot_logit, slot_gold, ignore_index=pad_id, reduction="mean"
    )
    intent_loss = nn.functional.cross_entropy(
      intent_logit, intent_gold, reduction="mean"
    )
    loss = slot_loss + intent_loss
    return loss

  def predict(self, b_subword_ids, b_masks):
    slot_logit, intent_logit = self._model(
      b_subword_ids.transpose(0,1), b_masks.transpose(0,1)
    )
    return slot_logit, intent_logit

  def evaluate_file(self, data_file):
    # data_file: '.pydict'
    start_time = time.time()
    label_tokenizer = LabelTokenizer.get_instance()
    slot_preds = []
    slot_golds = []
    intent_preds = []
    intent_golds = []
    for _, batch in get_batch_data([data_file], 1, False):
      batch = [e.to(self._device) for e in batch]
      b_subword_ids, b_masks, b_intent_ids, b_tag_ids = batch
      # b_masks: (b,l)
      # b_tag_ids: (b,l)
      # b_intent_ids: (b)
      slot_logit, intent_logit = self.predict(b_subword_ids, b_masks)
      slot_logit = slot_logit.view(-1, self._param.tgt_vocab_size) # (l*b,n_tgt)
      intent_logit = intent_logit.view(-1, self._param.n_intent)

      b_masks = b_masks.ne(True)

      slot_pred = \
        slot_logit.max(1)[1].view(self._param.max_seq_len, -1).transpose(0, 1) #(b,l)

      slot_pred = [
        [
          label_tokenizer.idx2tag[tag.item()]
          for tag in tags
        ]
        for tags in torch.mul(slot_pred, b_masks)
      ]

      slot_gold = [
        [
          label_tokenizer.idx2tag[tag.item()]
          for tag in tags
        ]
        for tags in torch.mul(b_tag_ids, b_masks)
      ]

      intent_pred = intent_logit.max(1)[1]  # (b)
      # b_intent_ids = b_intent_ids.view(-1)  # (b)
      intent_pred = [
        label_tokenizer.idx2intent[tag.item()] for tag in intent_pred
      ]
      intent_gold = [
        label_tokenizer.idx2intent[tag.item()] for tag in b_intent_ids
      ]

      slot_preds += slot_pred
      slot_golds += slot_gold
      intent_preds += intent_pred
      intent_golds += intent_gold

    lines = ''
    for intent_pred, slot_pred in zip(intent_preds, slot_preds):
      line = intent_pred + ' ' + ' '.join(slot_pred)
      lines += line + '\n'
    with open(f"{self._param.path_work}/pred.txt", 'w') as f:
      f.write(lines)

    lines = ''
    for intent_gold, slot_gold in zip(intent_golds, slot_golds):
      line = intent_gold + ' ' + ' '.join(slot_gold)
      lines += line + '\n'
    with open(f"{self._param.path_work}/gold.txt", 'w') as f:
      f.write(lines.strip())

    f1, precision, recall = compute_f1_score(slot_golds, slot_preds)
    intent_acc, sent_acc = get_sent_acc(
      f"{self._param.path_work}/gold.txt",
      f"{self._param.path_work}/pred.txt"
    )

    print('F1: {0:.3f}, Precision: {0:.3f}, Recall: {0:.3f}'.format(
      f1, precision, recall
    ))
    print('Intent: {0:.3f}'.format(intent_acc))
    print('Sent_acc: {0:.3f}'.format(sent_acc))
    return f1



def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  nlp.display_server_info()
  Logger.set_level(options.debug_level)

  trainer = Trainer()
  trainer.train()

if __name__ == "__main__":
  main()