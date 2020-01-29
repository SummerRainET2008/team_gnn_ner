#coding: utf8

import os
import sys
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import logging
import random
from tqdm import tqdm
import ver_1_elmo.constants as constants
from ver_1_elmo.dataset import Dataset, paired_collate_fn
from ver_1_elmo._model import sLSTM
from ver_1_elmo.evaluate import compute_f1_score, get_sent_acc

def cal_correct(pred, gold):
  '''
  Args:
    pred: pred_logits (b,emb)
  Returns: number of correct predictions
  '''
  pred = pred.max(1)[1]
  gold = gold.contiguous().view(-1)
  non_pad_mask = gold.ne(constants.PAD) # not equal
  n_correct = pred.eq(gold) # equal
  n_correct = n_correct.masked_select(non_pad_mask).sum().item()
  return n_correct

def train_epoch(model, training_data, optimizer, device):
  ''' Epoch operation in training phase '''

  model.train()

  total_loss = 0
  slot_total = 0
  slot_correct = 0
  intent_total = 0
  intent_correct = 0

  for batch in tqdm(
      training_data, mininterval=2, desc='  - (Training)   ', leave=False
  ):
    src_seq, src_char, tgt_seq = map(lambda x: x.to(device), batch)
    # src_seq, tgt_seq: [batch_size, max_len]
    # src_char: [batch_size, max_len, 50]
    slot_gold = tgt_seq[:, 1:].transpose(0, 1).contiguous().view(-1)
    intent_gold = tgt_seq[:, 0].view(-1)
    intent_gold[intent_gold != constants.UNK] -= constants.TAGS

    # first element in every sentence is '<s>'
    src_seq = src_seq[:, 1:]
    src_char = src_char[:, 1:]

    optimizer.zero_grad()
    slot_logit, intent_logit = model(src_seq.transpose(0, 1), src_char, )# [B*C]
    slot_loss = F.cross_entropy(
      slot_logit, slot_gold, ignore_index=constants.PAD, reduction='sum'
    )
    intent_loss = F.cross_entropy(intent_logit, intent_gold, reduction='sum')

    batch_loss = slot_loss + intent_loss
    # batch_loss = slot_loss
    # batch_loss = intent_loss

    batch_slot_correct = cal_correct(slot_logit, slot_gold)
    batch_intent_correct = cal_correct(intent_logit, intent_gold)
    batch_loss.backward()
    optimizer.step()
    # optimizer.step_and_update_lr()
    total_loss += batch_loss.item()
    slot_correct += batch_slot_correct
    intent_correct += batch_intent_correct
    non_pad_mask = slot_gold.ne(constants.PAD)
    batch_word = non_pad_mask.sum().item()
    # how many words in all sentences in this batch
    slot_total += batch_word
    intent_total += src_seq.size(0) # how many sentences

  loss_per_word = total_loss / slot_total
  slot_accuracy = slot_correct / slot_total
  intent_accuracy = intent_correct / intent_total
  return loss_per_word, slot_accuracy, intent_accuracy

def eval_f1(model, validation_data, device, opt):
  model.eval()
  data = torch.load(opt.data)
  tgt_word2idx = data['dict']['tgt']
  tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
  intent_word2idx = data['dict']['intent']
  intent_idx2word = {idx: word for word, idx in intent_word2idx.items()}
  slot_preds = []
  slot_golds = []
  intent_preds = []
  intent_golds = []
  with torch.no_grad():
    for batch in tqdm(
        validation_data, mininterval=2, desc='  - (computing F1) ', leave=False
    ):
      src_seq, src_char, tgt_seq = map(lambda x: x.to(device), batch) # [B,max_len]
      src_seq = src_seq[:, 1:]
      src_char = src_char[:, 1:]
      slot_gold = tgt_seq[:, 1:]
      max_len = src_seq.size(1)

      intent_gold = tgt_seq[:, 0].view(-1)
      intent_gold[intent_gold != constants.UNK] -= constants.TAGS

      slot_logit, intent_logit = model(src_seq.transpose(0, 1), src_char) # [B*C]
      intent_pred = intent_logit.max(1)[1]
      slot_pred = slot_logit.max(1)[1].view(max_len, -1).transpose(0, 1)
      slot_pred = [
        [
          tgt_idx2word.get(elem.item(), constants.UNK_WORD)
         for elem in elems if elem.item() != constants.PAD
        ]
        for elems in slot_pred
      ]
      slot_gold = [
        [
          tgt_idx2word.get(elem.item(), constants.UNK_WORD)
         for elem in elems if elem.item() != constants.PAD
        ]
        for elems in slot_gold
      ]

      intent_pred = [
        intent_idx2word.get(elem.item() + constants.TAGS, constants.UNK_WORD)
        for elem in intent_pred
      ]
      intent_gold = [
        intent_idx2word.get(elem.item() + constants.TAGS, constants.UNK_WORD)
        for elem in intent_gold
      ]

      for i in range(len(slot_gold)):
        slot_pred[i] = slot_pred[i][:len(slot_gold[i])]
      slot_preds += slot_pred
      slot_golds += slot_gold
      intent_preds += intent_pred
      intent_golds += intent_gold

  lines = ''
  for intent_pred, slot_pred in zip(intent_preds, slot_preds):
    line = intent_pred + ' ' + ' '.join(slot_pred)
    lines += line + '\n'
  with open('pred.txt', 'w') as f:
    f.write(lines)

  lines = ''
  for intent_gold, slot_gold in zip(intent_golds, slot_golds):
    line = intent_gold + ' ' + ' '.join(slot_gold)
    lines += line + '\n'
  with open('gold.txt', 'w') as f:
    f.write(lines.strip())

  f1, precision, recall = compute_f1_score(slot_golds, slot_preds)
  intent_acc, sent_acc = get_sent_acc('gold.txt', 'pred.txt')

  print(' F1: {0:.3f}, Precision: {0:.3f}, Recall: {0:.3f}'.format(
    f1, precision, recall
  ))
  print(' Intent: {0:.3f}'.format(intent_acc))
  print(' Sent_acc: {0:.3f}'.format(sent_acc))
  return f1, precision, recall, intent_acc, sent_acc

def eval_epoch(model, validation_data, device):
  ''' Epoch operation in training phase'''

  model.eval()

  total_loss = 0
  slot_total = 0
  slot_correct = 0
  intent_total = 0
  intent_correct = 0

  for batch in tqdm(
      validation_data, mininterval=2, desc='  - (Validation)   ', leave=False
  ):
    src_seq, src_char, tgt_seq = map(lambda x: x.to(device), batch)  # [B,max_len]
    src_seq = src_seq[:, 1:]
    src_char = src_char[:, 1:]
    slot_gold = tgt_seq[:, 1:].transpose(0, 1).contiguous().view(-1)
    intent_gold = tgt_seq[:, 0].view(-1)
    intent_gold[intent_gold != constants.UNK] -= constants.TAGS

    slot_logit, intent_logit = model(src_seq.transpose(0, 1), src_char)  # [B*C]
    slot_loss = F.cross_entropy(
      slot_logit, slot_gold, ignore_index=constants.PAD, reduction='sum'
    )
    intent_loss = F.cross_entropy(intent_logit, intent_gold, reduction='sum')

    batch_loss = slot_loss + intent_loss

    batch_slot_correct = cal_correct(slot_logit, slot_gold)
    batch_intent_correct = cal_correct(intent_logit, intent_gold)

    total_loss += batch_loss.item()
    slot_correct += batch_slot_correct
    intent_correct += batch_intent_correct
    non_pad_mask = slot_gold.ne(constants.PAD)
    batch_word = non_pad_mask.sum().item()
    slot_total += batch_word
    intent_total += src_seq.size(0)

  loss_per_word = total_loss / slot_total
  slot_accuracy = slot_correct / slot_total
  intent_accuracy = intent_correct / intent_total
  return loss_per_word, slot_accuracy, intent_accuracy

def train(
    model, training_data, validation_data, test_data, optimizer, device, opt
):
  ''' Start training '''

  log_train_file = None
  log_valid_file = None

  if opt.log:
    log_train_file = opt.log + '.train.log'
    log_valid_file = opt.log + '.valid.log'

    print('[Info] Training performance will be written to file: '
          '{} and {}'.format(log_train_file, log_valid_file))

    with open(log_train_file, 'a') as log_tf, \
         open(log_valid_file, 'a') as log_vf:
      log_tf.write('\n')
      log_tf.write('epoch,loss,ppl,accuracy\n')
      log_vf.write('epoch,loss,ppl,accuracy\n')

  valid_accus = []
  f1s = []
  best_slot = 0
  best_intent = 0
  best_sent = 0
  for epoch_i in range(opt.epoch):
    print('[ Epoch', epoch_i, ']')

    start = time.time()
    train_loss, train_accu, train_intent_accu = train_epoch(
      model, training_data, optimizer, device,
    )  # the train_loss is per word loss

    print(
      '- (Training)   ppl: {ppl: 8.5f}, slot_accuracy: {accu:3.3f}%, ' \
      'intent_accuracy:{accu2:3.3f}%, ' \
      'elapse: {elapse:3.3f} min'.format(
        ppl=math.exp(min(train_loss, 100)),
        accu=100 * train_accu,
        accu2=100 * train_intent_accu,
        elapse=(time.time() - start) / 60)
    )

    start = time.time()
    valid_loss, valid_accu, valid_intent_accu = eval_epoch(
      model, validation_data, device
    )

    print(
      '- (Validation) ppl: {ppl: 8.5f}, valid_accuracy: {accu:3.3f}%, ' \
      'intent_accuracy:{accu2:3.3f}%, ' \
      'elapse: {elapse:3.3f} min'.format(
        ppl=math.exp(min(valid_loss, 100)),
        accu=100 * valid_accu,
        accu2=100 * valid_intent_accu,
        elapse=(time.time() - start) / 60)
    )
    start = time.time()
    test_loss, test_accu, test_intent_accu = eval_epoch(
      model, test_data, device
    )

    print(
      '- (Test)       ppl: {ppl: 8.5f}, test_accuracy: {accu:3.3f}%, ' \
      'intent_accuracy:{accu2:3.3f}%, ' \
      'elapse: {elapse:3.3f} min'.format(
        ppl=math.exp(min(test_loss, 100)),
        accu=100 * test_accu,
        accu2=100 * test_intent_accu,
        elapse=(time.time() - start) / 60)
    )

    f1, precision, recall, intent_acc, sent_acc = eval_f1(
      model, test_data, device, opt
    )
    f1s += [f1]

    if sent_acc > best_sent:
      best_slot = f1
      best_intent = intent_acc
      best_sent = sent_acc
    print(
      'Current best results $$ F1:{0:.3f}, Intent acc:{0:.3f}, ' \
      'Sent acc:{0:.3f} '.format(
        best_slot,best_intent,best_sent
      )
    )

    valid_accus.append(valid_accu)
    # scheduler.step(f1)

    if opt.parallel:
      model_state_dict = model.module.state_dict()
    else:
      model_state_dict = model.state_dict()

    checkpoint = {'model': model_state_dict, 'settings': opt, 'epoch': epoch_i}
    if opt.save_model:
      if opt.save_mode == 'all':
        model_name = opt.save_model + \
                     '_accu_{accu:3.3f}.chkpt'.format(accu=100 * f1)
        torch.save(checkpoint, model_name)
      elif opt.save_mode == 'best':
        model_name = opt.save_model
        # if valid_accu >= max(valid_accus):
        if f1 >= max(f1s):
          torch.save(checkpoint, model_name)
          print(
            '- [Info] The checkpoint file has been updated.-------------------')

    if log_train_file and log_valid_file:
      with open(log_train_file, 'a') as log_tf, \
           open(log_valid_file, 'a') as log_vf:
        log_tf.write(
          '{epoch},{loss: 8.5f},{ppl: 8.5f},'\
          '{accu:3.3f}\n'.format(epoch=epoch_i,
                                 loss=train_loss,
                                 ppl=math.exp(min(train_loss,100)),
                                 accu=100 * train_accu))
        log_vf.write(
          '{epoch},{loss: 8.5f},{ppl: 8.5f},' \
          '{accu:3.3f}\n'.format(epoch=epoch_i,
                                 loss=valid_loss,
                                 ppl=math.exp(min(valid_loss,100)),
                                 accu=100 * valid_accu))


def main():
  ''' Main function '''
  parser = argparse.ArgumentParser()
  parser.add_argument('-epoch', type=int, default=400)
  parser.add_argument('-batch_size', type=int, default=32)
  parser.add_argument('-d_model', type=int, default=150)
  # when using Elmo, d_word is set as 1024 in models/_model.py line 337
  # when using Embedding layer, d_word should be set here
  parser.add_argument('-d_word', type=int, default=1024)
  # parser.add_argument('-d_char', type=int, default=30)
  # parser.add_argument('-d_pos', type=int, default=512)
  # parser.add_argument('-d_inner_hid', type=int, default=1200)  # orginal 2048
  parser.add_argument('-dropout', type=float, default=0.5)
  parser.add_argument('-data', default='data/snips.pt', required=False)
  parser.add_argument('-log', default='logs/snips')
  parser.add_argument('-save_model', default='snips.chkpt')
  parser.add_argument('-save_mode', type=str, choices=['all', 'best'],
                      default='best')
  parser.add_argument('-restore_model', default=None) # 'atis.chkpt'
  parser.add_argument('-parallel', default=False)
  parser.add_argument('-no_cuda', default=True, action='store_true')
  # parser.add_argument('-n_warmup_steps', type=int, default=4000)
  # parser.add_argument('-label_smoothing', action='store_true')

  opt = parser.parse_args()
  opt.cuda = not opt.no_cuda

  os.environ["CUDA_VISIBLE_DEVICES"] = constants.GPU
  logging.disable(sys.maxsize)  # Python 3

  np.random.seed(constants.SEED)
  torch.manual_seed(constants.SEED)
  torch.cuda.manual_seed(constants.SEED)
  random.seed(constants.SEED)
  torch.backends.cudnn.deterministic = True

  # ========= Loading Dataset =========#
  data = torch.load(opt.data)
  opt.max_token_seq_len = data['settings'].max_token_seq_len
  training_data, validation_data, test_data = prepare_dataloaders(data, opt)
  opt.src_vocab_size = training_data.dataset.src_vocab_size
  opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

  # ========= Preparing Model =========#
  print(opt)
  device = torch.device('cuda' if opt.cuda else 'cpu')
  embeddings = None
  opt.n_intent = len(data['dict']['intent'])
  opt.n_tgt = len(data['dict']['tgt'])
  model = sLSTM(
    d_word=opt.d_word, d_hidden=opt.d_model,
    n_src_vocab=opt.src_vocab_size, n_tgt_vocab=opt.tgt_vocab_size,
    n_intent=opt.n_intent, dropout=opt.dropout,
    embeddings=embeddings
  ).to(device)
  if opt.restore_model:
    checkpoint = torch.load(opt.restore_model)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Old Trained model state loaded.')

  if opt.parallel:
    model = nn.DataParallel(model)
  optimizer = optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    betas=(0.9, 0.98), eps=1e-09
  )
  # optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
  #                      lr=0.001,momentum=0.9)

  train(model, training_data, validation_data, test_data, optimizer, device, opt)


def prepare_dataloaders(data, opt):
  train_loader = torch.utils.data.DataLoader(
    Dataset(src_word2idx=data['dict']['src'], tgt_word2idx=data['dict']['tgt'],
                       src_insts=data['train']['src'],
                       src_char_insts=data['train']['src_char'],
                       tgt_insts=data['train']['tgt']),
    num_workers=2,
    batch_size=opt.batch_size,
    collate_fn=paired_collate_fn, shuffle=True)

  valid_loader = torch.utils.data.DataLoader(
    Dataset(src_word2idx=data['dict']['src'], tgt_word2idx=data['dict']['tgt'],
                       src_insts=data['valid']['src'],
                       src_char_insts=data['valid']['src_char'],
                       tgt_insts=data['valid']['tgt']),
    num_workers=2,
    batch_size=opt.batch_size,
    collate_fn=paired_collate_fn)
  test_loader = torch.utils.data.DataLoader(
    Dataset(src_word2idx=data['dict']['src'], tgt_word2idx=data['dict']['tgt'],
                       src_insts=data['test']['src'],
                       src_char_insts=data['test']['src_char'],
                       tgt_insts=data['test']['tgt']),
    num_workers=2,
    batch_size=opt.batch_size,
    collate_fn=paired_collate_fn)
  return train_loader, valid_loader, test_loader

if __name__ == '__main__':
  main()
