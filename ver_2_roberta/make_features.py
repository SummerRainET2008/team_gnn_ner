#coding: utf8

from ver_2_roberta import *
from ver_2_roberta.param import Param
from pytorch_transformers import RobertaTokenizer
from pa_nlp.nlp import *


class LabelTokenizer:
  _inst = None

  def __init__(self):
    param = Param()
    label_insts = []
    with open(param.file_for_label_tokenizer) as f:
      for sent in f:
        # sent = sent.lower()
        words = sent.split()
        label_insts += [words]

    full_intent = sorted(
      set(w for inst in label_insts for i, w in enumerate(inst) if i == 0)
    )
    full_tag = sorted(
      set(w for inst in label_insts for i, w in enumerate(inst) if i != 0)
    )

    tag2idx = {
      BOS_WORD: cls_id,
      EOS_WORD: sep_id,
      PAD_WORD: pad_id,
      UNK_WORD: unk_id
    }
    intent2idx = {UNK_WORD: unk_id}

    for i in full_intent:
      if i not in intent2idx:
        intent2idx[i] = len(intent2idx)
    self.idx2intent = {value: key for key, value in intent2idx.items()}

    for i in full_tag:
      if i not in tag2idx:
        tag2idx[i] = len(tag2idx)
    self.idx2tag = {value: key for key, value in tag2idx.items()}

    self.indent2idx = intent2idx
    self.tag2idx = tag2idx
    print(f'number of intents:{len(self.indent2idx)}')
    print(f'number of tags:{len(self.tag2idx)}')
    print(f'O:{self.tag2idx["O"]}')

  @staticmethod
  def get_instance():
    if LabelTokenizer._inst is None:
      LabelTokenizer._inst = LabelTokenizer()
    return LabelTokenizer._inst

  def tokenize(self, label, max_len):
    # TODO: only make tag which makes KeyError as unk
    # label = label.lower().split()
    label = label.split()
    try:
      intent_id = self.indent2idx[label[0]]
      tag_ids = [self.tag2idx[x] for x in label[1:]]
      tag_ids = tag_ids[: max_len]
      diff = max_len - len(tag_ids)
      tag_ids = tag_ids + [pad_id] * diff
    except KeyError:
      print(f'KeyError: {label}')
      intent_id = unk_id
      tag_ids = [unk_id] * max_len

    return intent_id, tag_ids

class Tokenizer:
  _inst = None

  def __init__(self):
    param = Param()
    inst = RobertaTokenizer.from_pretrained(
        param.pretrained_model
    )
    self._tokenizer = inst
    self._cls_idx = inst.convert_tokens_to_ids(inst.cls_token)
    self._sep_idx = inst.convert_tokens_to_ids(inst.sep_token)
    self._pad_idx = inst.convert_tokens_to_ids(inst.pad_token)
    print(f"cls_id: {self._cls_idx}")  # 0
    print(f"sep_id: {self._sep_idx}")  # 2
    print(f"pad_id: {self._pad_idx}")  # 1

  @staticmethod
  def get_instance():
    if Tokenizer._inst is None:
      Tokenizer._inst = Tokenizer()
    return Tokenizer._inst

  def tokenize(self, sentence, max_len):
    sentence = sentence.lower()
    ids = self._tokenizer.encode(sentence)
    word_ids = ids[1:]
    word_ids = word_ids[: max_len]
    diff = max_len - len(word_ids)
    word_ids = word_ids + [self._pad_idx] * diff
    mask = [id == self._pad_idx for id in word_ids]

    return word_ids, mask

  def convert_ids(self, subword_ids):
    subword_list = self._tokenizer.convert_ids_to_tokens(subword_ids)
    result = []
    for subword in subword_list:
      if subword in (PAD_WORD, BOS_WORD, EOS_WORD):
        continue
      else:
        if 'Ġ' in subword:
          subword = subword[1:]
      result.append(subword)
    return result

  def decode(self, subword_ids):
    sent = self._tokenizer.decode(subword_ids)
    return sent

  def match(self, word_list, subword_ids):
    indices = []
    tmp = []
    count = 0
    j = 0
    for i in range(len(word_list)):
      if self.decode(subword_ids[j]).strip() == word_list[i]:
        indices.append(i)
      else:
        while self.decode(tmp + [subword_ids[j]]).strip() != word_list[i]:
          tmp += [subword_ids[j]]
          j += 1
          count += 1
        indices.extend([i] * (count + 1))
        tmp = []
        count = 0
      j += 1
    return indices

  def get_vob_size(self):
    return len(self._tokenizer)


def process(src_file: str, tgt_file: str, out_file: str, param: Param):
  def get_point(file):
    for ln in next_line_from_file(file):
      point = ln.strip()
      yield point

  def data_generator():
    tokenizer = Tokenizer.get_instance()
    label_tokenizer = LabelTokenizer.get_instance()
    src_insts = []
    indices_insts = []
    masks = []
    intent_insts = []
    tag_insts = []
    skip_lines = []
    line = 0
    for sent in get_point(src_file):
      line += 1
      subword_ids, mask = tokenizer.tokenize(sent, param.max_seq_len)
      # subword_list = tokenizer.convert_ids(subword_ids)
      # print(subword_list)
      try:
        indices = tokenizer.match(sent.lower().split()[1:], subword_ids)
      except IndexError:
        skip_lines.append(line)
        print(f'IndexError: {sent}')
        continue
      src_insts.append(subword_ids)
      masks.append(mask)
      indices_insts.append(indices)

    line = 0
    for label in get_point(tgt_file):
      line += 1
      if line in skip_lines:
        continue
      intent_id, tag_ids = label_tokenizer.tokenize(label, param.max_seq_len)
      intent_insts.append(intent_id)
      tag_insts.append(np.array(tag_ids))

    # print(len(src_insts), len(indices_insts), len(masks),
    #       len(intent_insts), len(tag_insts))
    assert len(src_insts) == len(indices_insts) == len(masks)\
           == len(intent_insts) == len(tag_insts)
    print('[INFO] {} samples in total.'.format(len(src_insts)))
    for (subword_ids, indices, mask, intent_id, tag_ids) \
      in zip(src_insts, indices_insts, masks, intent_insts, tag_insts):
      # print(f'tag_id0:{tag_ids}')
      tag_ids = list(tag_ids[indices])
      tag_ids = tag_ids[: param.max_seq_len]
      diff = param.max_seq_len - len(tag_ids)
      tag_ids = tag_ids + [pad_id] * diff

      # print(f'tag_id1:{tag_ids}')
      yield subword_ids, mask, intent_id, tag_ids

  data = list(data_generator())
  pickle.dump(data, open(out_file, "wb"))

def main():
  parser = optparse.OptionParser(usage="cmd [optons]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--log_level", type=int, default=1)
  (options, args) = parser.parse_args()
  Logger.set_level(options.log_level)

  param = Param()

  print('[INFO] Writing training files.')
  process(
    "data/atis/train/intent_seq.in", "data/atis/train/intent_seq.out",
    f"{param.path_feat}/train.pydict", param
  )
  print('[INFO] Writing validation files.')
  process(
    "data/atis/train/intent_seq.in", "data/atis/train/intent_seq.out",
    f"{param.path_feat}/vali.pydict", param
  )
  # print('[INFO] Writing test files.')
  # process(
  #   "data/snips/train/intent_seq.in", "data/snips/train/intent_seq.out",
  #   f"{param.path_feat}/test.pydict", param
  # )

if __name__ == "__main__":
  main()
