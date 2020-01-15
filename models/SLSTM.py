#coding: utf8

import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Constants
from utils import get_len
from allennlp.modules.elmo import Elmo
from \
  allennlp.modules.seq2seq_encoders.stacked_self_attention \
  import \
  StackedSelfAttentionEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = Constants.GPU

class penalized_tanh(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, tensor):
    return 0.75 * F.relu(F.tanh(tensor)) + 0.25 * F.tanh(tensor)

class sLSTMCell(nn.Module):
  '''
  Args:
      input_size: feature size of input sequence
      hidden_size: size of hidden state
      window_size: size of context window
      sentence_nodes:
      bias: Default: ``True``
      batch_first: default False follow the pytorch convenient
      dropout:  Default: 0
      initial_mathod: 'orgin' for pytorch default

  Inputs: (input, length), (h_0, c_0)
      --input: (seq_len, batch, input_size)
      --length: (batch, 1)
      --h_0: (seq_len+sentence_nodes, batch, hidden_size)
      --c_0: (seq_len+sentence_nodes, batch, hidden_size)
  Outputs: (h_1, c_1)
      --h_1: (seq_len+sentence_nodes, batch, hidden_size)
      --c_1: (seq_len+sentence_nodes, batch, hidden_size)

  '''

  def __init__(self, d_word, d_hidden, n_windows=1,
               n_sent_nodes=1, bias=True, batch_first=False,
               init_method='normal'):
    super().__init__()
    self.d_input = d_word
    self.d_hidden = d_hidden
    self.n_windows = n_windows
    self.num_g = n_sent_nodes
    self.initial_method = init_method
    self.bias = bias
    self.ptanh = penalized_tanh()
    # self.batch_first = batch_first
    # self.lens_dim = 1 if batch_first is True else 0
    self._all_gate_weights = []

    # define parameters for word nodes
    word_gate_dict = dict(
      [('input_gate', 'i'), ('left_forget_gate', 'l'),
       ('right_forget_gate', 'r'), ('forget_gate', 'f'),
       ('sentence_forget_gate', 's'), ('output_gate', 'o'),
       ('recurrent_input', 'u')])

    for (gate_name, gate_tag) in word_gate_dict.items():
      # weight: (out_features, in_features)
      w_w = nn.Parameter(torch.Tensor(d_hidden,
                                      (n_windows * 2 + 1) * d_hidden))
      w_u = nn.Parameter(torch.Tensor(d_hidden, d_word))
      w_v = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
      w_b = nn.Parameter(torch.Tensor(d_hidden))

      gate_params = (w_w, w_u, w_v, w_b)
      param_names = ['w_w{}', 'w_u{}', 'w_v{}', 'w_b{}']
      param_names = [x.format(gate_tag) for x in param_names]
      for name, param in zip(param_names, gate_params):
        setattr(self, name, param)  # self.w_w{i} = w_w
      self._all_gate_weights.append(param_names)

    # define parameters for sentence node
    sentence_gate_dict = dict(
      [('sentence_forget_gate', 'g'), ('word_forget_gate', 'f'),
       ('output_gate', 'o')])

    for (gate_name, gate_tag) in sentence_gate_dict.items():
      # weight: (out_features, in_features)
      s_w = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
      s_u = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
      s_b = nn.Parameter(torch.Tensor(d_hidden))
      gate_params = (s_w, s_u, s_b)
      param_names = ['s_w{}', 's_u{}', 's_b{}']
      param_names = [x.format(gate_tag) for x in param_names]
      for name, param in zip(param_names, gate_params):
        setattr(self, name, param)  # self.s_w{i} = s_w
      self._all_gate_weights.append(param_names)

    self.reset_parameters(self.initial_method)

  def reset_parameters(self, init_method):
    if init_method is 'normal':
      std = 0.1
      for weight in self.parameters():
        weight.data.normal_(mean=0.0, std=std)
    else:  # uniform: make std of weights as 0
      stdv = 1.0 / math.sqrt(self.d_hidden)
      for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)

  def sequence_mask(self, size, length):
    ''' Padding positions are 1 for masked_fill func to use '''
    mask = torch.LongTensor(range(size[0])).view(size[0], 1)  # (l,1)
    # mask = mask.cuda()
    length = length.squeeze(dim=1)  # (b)
    result = (mask >= length).unsqueeze(dim=2)  # (l,b,1)
    return result

  def in_window_context(self, hx, window_size=1):
    '''

    Args:
        hx: (l,b,d)
        window_size:

    Returns: (l,b,hidden*(window*2+1)

    '''

    slices = torch.unbind(hx, dim=0)
    # torch.size([18,32,256]) -> ([32,256]) * 18
    zeros = torch.unbind(torch.zeros_like(hx), dim=0)

    context_l = [torch.stack(zeros[:i] + slices[:len(slices) - i], dim=0)
                 for i in range(window_size, 0, -1)]
    context_l.append(hx)
    context_r = [
      torch.stack(slices[i + 1: len(slices)] + zeros[:i + 1], dim=0)
      for i in range(0, window_size)]

    context = context_l + context_r
    # context is a list of length window size*2+1,
    # every element covering different part of original sent,
    # every element in context is of same length
    return torch.cat(context, dim=2)

  def forward(self, src_seq, src_len, state=None):
    # src_seq here is actually embedded src_char when using Elmo

    seq_mask = self.sequence_mask(src_seq.size(), src_len)
    h_gt_1 = state[0][-self.num_g:]  # sent node is in the end
    h_wt_1 = state[0][:-self.num_g].masked_fill(seq_mask, 0) #（l,b,d)
    c_gt_1 = state[1][-self.num_g:]
    c_wt_1 = state[1][:-self.num_g].masked_fill(seq_mask, 0)

    # update sentence node
    h_hat = h_wt_1.mean(dim=0)
    fg = F.sigmoid(F.linear(h_gt_1, self.s_wg) +
                   F.linear(h_hat, self.s_ug) +
                   self.s_bg)
    o = F.sigmoid(F.linear(h_gt_1, self.s_wo) +
                  F.linear(h_hat, self.s_uo) + self.s_bo)
    fi = F.sigmoid(F.linear(h_gt_1, self.s_wf) +
                   F.linear(h_wt_1, self.s_uf) +
                   self.s_bf).masked_fill(seq_mask, -1e25)
    fi_normalized = F.softmax(fi, dim=0)
    c_gt = fg.mul(c_gt_1).add(fi_normalized.mul(c_wt_1).sum(dim=0))
    h_gt = o.mul(F.tanh(c_gt))

    # update word nodes
    epsilon = self.in_window_context(h_wt_1, window_size=self.n_windows)
    # epsilon: (l, b, d_word or emb_size * (2 * window_size + 1)
    i = F.sigmoid(F.linear(epsilon, self.w_wi) +
                  F.linear(src_seq, self.w_ui) +
                  F.linear(h_gt_1, self.w_vi) + self.w_bi)
    l = F.sigmoid(F.linear(epsilon, self.w_wl) +
                  F.linear(src_seq, self.w_ul) +
                  F.linear(h_gt_1, self.w_vl) + self.w_bl)
    r = F.sigmoid(F.linear(epsilon, self.w_wr) +
                  F.linear(src_seq, self.w_ur) +
                  F.linear(h_gt_1, self.w_vr) + self.w_br)
    f = F.sigmoid(F.linear(epsilon, self.w_wf) +
                  F.linear(src_seq, self.w_uf) +
                  F.linear(h_gt_1, self.w_vf) + self.w_bf)
    s = F.sigmoid(F.linear(epsilon, self.w_ws) +
                  F.linear(src_seq, self.w_us) +
                  F.linear(h_gt_1, self.w_vs) + self.w_bs)
    o = F.sigmoid(F.linear(epsilon, self.w_wo) +
                  F.linear(src_seq, self.w_uo) +
                  F.linear(h_gt_1, self.w_vo) + self.w_bo)
    u = F.tanh(F.linear(epsilon, self.w_wu) +
               F.linear(src_seq, self.w_uu) +
               F.linear(h_gt_1, self.w_vu) + self.w_bu)

    gates = torch.stack((l, f, r, s, i), dim=0) # (5*l,b,d)
    gates_normalized = F.softmax(gates.masked_fill(seq_mask, -1e25), dim=0)

    c_wt_l, c_wt_1, c_wt_r = \
      self.in_window_context(c_wt_1).chunk(3, dim=2) # split by dim 2
    # c_wt_: (l, b, d_word)
    c_mergered = torch.stack((c_wt_l, c_wt_1, c_wt_r,
                              c_gt_1.expand_as(c_wt_1.data), u), dim=0)

    c_wt = gates_normalized.mul(c_mergered).sum(dim=0)
    c_wt = c_wt.masked_fill(seq_mask, 0)
    h_wt = o.mul(F.tanh(c_wt))

    h_t = torch.cat((h_wt, h_gt), dim=0)
    c_t = torch.cat((c_wt, c_gt), dim=0)
    return (h_t, c_t)

class sLSTM(nn.Module):
  '''
  Args:
      input_size: feature size of input sequence
      hidden_size: size of hidden sate
      window_size: size of context window
      steps: num of iteration step
      sentence_nodes:
      bias: use bias if is True
      batch_first: default False follow the pytorch convenient
      dropout: elements are dropped by this probability, default 0

  Inputs: (src_seq,src_len, state
      --src_seq: (seq_len, batch, input_size)
      --src_len: (batch, 1)
      --state[0], h_0: (seq_len+sentence_nodes, batch, hidden_size)
      --state[1], c_0: (seq_len+sentence_nodes, batch, hidden_size)
  Outputs: h_t, g_t
      --h_t: (seq_len, batch, hidden_size), output of every word in inputs
      --g_t: (sentence_nodes, batch, hidden_size),
          output of sentence node
  '''

  def __init__(self, d_word, n_src_vocab, n_tgt_vocab, n_intent, d_hidden,
               window_size=3, steps=4, sentence_nodes=1, bias=True,
               batch_first=False, dropout=0.2, embeddings=None):
    super(sLSTM, self).__init__()
    self.sw1 = nn.Sequential(nn.Conv1d(
      d_hidden, d_hidden, kernel_size=1, padding=0),
      nn.BatchNorm1d(d_hidden), nn.ReLU())
    self.sw3 = nn.Sequential(nn.Conv1d(
      d_hidden, d_hidden, kernel_size=1, padding=0),
      nn.ReLU(), nn.BatchNorm1d(d_hidden),
      nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
      nn.ReLU(), nn.BatchNorm1d(d_hidden))
    self.sw33 = nn.Sequential(
      nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
      nn.ReLU(), nn.BatchNorm1d(d_hidden),
      nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
      nn.ReLU(), nn.BatchNorm1d(d_hidden))

    self.filter_linear = nn.Linear(3 * d_hidden, d_hidden)
    self.multi_att = StackedSelfAttentionEncoder(input_dim=d_hidden,
                                                 hidden_dim=d_hidden,
                                                 projection_dim=d_hidden,
                                                 feedforward_hidden_dim
                                                               =2 * d_hidden,
                                                 num_layers=1,
                                                 num_attention_heads=5)

    self.steps = steps
    self.d_hidden = d_hidden
    self.sentence_nodes = sentence_nodes
    self.embeddings = nn.Embedding(n_src_vocab, d_word)
    self.n_tgt_vocab = n_tgt_vocab
    self.n_intent = n_intent
    self.elmo = Elmo(Constants.ELMO_OPTIONS, Constants.ELMO_WEIGHT, 1,
                     requires_grad=False, dropout=dropout)
    self.slot_out = nn.Linear(d_hidden, n_tgt_vocab)
    self.intent_out = nn.Linear(d_hidden, n_intent)
    self.dropout = nn.Dropout(dropout)
    self.cell = sLSTMCell(d_word=d_word, d_hidden=d_hidden,
                          n_windows=window_size,
                          n_sent_nodes=sentence_nodes, bias=bias,
                          batch_first=batch_first)
    self.sigmoid = nn.Sigmoid()

  def _get_conv(self, src):
    '''

    Args:
        src: (l,b,d)

    Returns: (l,b,d)

    '''
    old = src
    src = src.transpose(0, 1).transpose(1, 2)  # (l,b,d) ->(b,l,d) ->(b,d,l)
    conv1 = self.sw1(src)
    conv3 = self.sw3(src)
    conv33 = self.sw33(src)
    conv = torch.cat([conv1, conv3, conv33], dim=1)  # (b,3d,l)
    conv = self.filter_linear(conv.transpose(1, 2)).transpose(0, 1)
    # (b,3d,l)->(b,l,3d)-> (b,l,d) -> (l,b,d)
    conv += old
    return conv

  def _get_self_attn(self, src, mask):
    '''

    Args:
        src: (l,b,d)
        mask:

    Returns: (l,b,d)

    '''
    attn = self.multi_att(src, mask)
    attn += src
    return attn

  def forward(self, src_seq, src_char, state=None):
    '''

    Args:
        src_seq: (l,b) for create mask when Elmo
        src_char: (b,l,50) for elmo embedding
        state:

    Returns:

    '''

    mask = src_seq.gt(Constants.PAD)
    # src_len = torch.cuda.LongTensor(
    #     np.array(get_len(src_seq.cpu().transpose(0, 1)))).unsqueeze(1)
    src_len = torch.LongTensor(
      np.array(get_len(src_seq.cpu().transpose(0, 1)))).unsqueeze(1)

    # Elmo embedding
    elmo_embs = self.elmo(src_char)['elmo_representations'][0]  # (b,l,d)
    elmo_embs = elmo_embs.transpose(0, 1)
    src_embs = elmo_embs # (l,b,d)

    # word embedding
    # src_embs = self.embeddings(src_seq)
    # src_embs = self.dropout(src_embs)

    if state is None:
      h_t = torch.zeros(src_embs.size(0) + self.sentence_nodes,
                        src_embs.size(1),
                        self.d_hidden)
      # h_t = h_t.cuda()
      c_t = torch.zeros_like(h_t)
    else:
      h_t = state[0]
      c_t = state[1]

    for step in range(self.steps):
      h_t, c_t = self.cell(src_embs, src_len, (h_t, c_t))

    h_t = self.dropout(h_t)
    h_w = h_t[:-self.sentence_nodes]  # (l,b,d)
    h_s = h_t[-self.sentence_nodes:]

    # attn-gate
    # attn = self._get_self_attn(h_w,mask)
    # attn_gate = self.sigmoid(attn)
    # h_w *= attn_gate

    # conv-attn-gate
    conv = self._get_conv(h_w)
    attn = self._get_self_attn(conv, mask)
    attn_gate = self.sigmoid(attn)
    h_w *= attn_gate

    slot_logit = self.slot_out(h_w).view(-1, self.n_tgt_vocab) # (l,b,n_tgt)
    sent_logit = self.intent_out(h_s).view(-1, self.n_intent)  # (1,b,n_intent)
    return slot_logit, sent_logit
