from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk
# import stanfordnlp


def del_intent(f_label, f_nlabel):
    with open(f_label) as f_in, open(f_nlabel, 'w') as f_out:
        for line in f_in:
            if '#' in line:
                line = line.split('#')[0]
            f_out.write(line)


def cnt_intent(f_intent):
    intents = set()
    with open(f_intent, 'r') as intent_file:
        for line in intent_file:
            intents.add(line.strip())
    print('intents:', len(intents))


def check_pos(f_src, f_pos):
    with open(f_pos) as f_pos, open(f_src) as f_in:
        for src, pos in zip(f_in, f_pos):
            if len(src.split()) != len(pos.split()):
                print(src)
                print(pos)


def capitalize_NNP_and_get_POS(f_src, f_cap, f_pos):
    nlp = stanfordnlp.Pipeline(processors='tokenize,pos,lemma', tokenize_pretokenized=True)

    with open(f_src) as f_src, open(f_pos, 'w') as f_pos, open(f_cap, 'w') as f_cap:
        for line in f_src:
            doc = nlp(line)
            sent = doc.sentences
            words = sent[0].words
            nl_line = list(map(lambda x: x.text.capitalize() if x.xpos == 'NNP' else x.text, words))
            n_line = ' '.join(nl_line)
            f_cap.write(n_line + '\n')
            posl_line = [word.xpos for word in words]
            pos_line = ' '.join(posl_line)
            f_pos.write(pos_line + '\n')


def get_lemma(f_cap, f_lemma):
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,lemma')
    with open(f_cap) as f_cap, open(f_lemma, 'w') as f_lemma:
        for line in f_cap:
            doc = nlp(line)
            sent = doc.sentences
            words = sent[0].words
            lemmal_line = [word.lemma for word in words]
            lemma_line = ' '.join(lemmal_line)
            f_lemma.write(lemma_line + '\n')


def get_cap_fea(f_cap, f_fea):
    with open(f_cap, 'r') as f_cap, open(f_fea, 'w') as f_fea:
        for line in f_cap:
            fea = []
            for word in line.split():
                if word.istitle():
                    fea.append('1')
                else:
                    fea.append('0')
            fea = ' '.join(fea)
            print(fea)
            f_fea.write(fea + '\n')

# def get_uni_intent(f_label, f_uni):
#     with open(f_label) as f_label, open(f_uni, 'w') as f_uni:
#         for line in f_label:
#             if '#' in line:
#                 f_uni.write(line.split('#')[0] + '\n')
#             else:
#                 f_uni.write(line)

def get_uni_intent(f_label, f_in, f_out, f_in_tgt, f_out_tgt):
    with open(f_label) as f_label, open(f_in) as f_in, open(f_in_tgt, 'w') as f_in_tgt, open(f_out) as f_out, open(
            f_out_tgt, 'w') as f_out_tgt:
        for (l_label, l_in, l_out) in zip(f_label, f_in, f_out):
            l_in = l_in.split()
            l_in = list(map(lambda x: 'DIGIT' *len(x) if x.isdigit() else x, l_in))
            l_in = ' '.join(l_in) +'\n'

            # if '#' in l_label:
            #     f_out_tgt.write(l_label.split('#')[0].strip() +' '+ l_out)
            #     f_out_tgt.write(l_label.split('#')[1].strip() +' ' + l_out)
            #     f_in_tgt.write('<s>' + ' ' + l_in)
            #     f_in_tgt.write('<s>' + ' ' + l_in)
            # else:
            f_out_tgt.write(l_label.strip() + ' ' + l_out)
            f_in_tgt.write('<s>' + ' ' + l_in)


def get_pad(f_in, f_out):
    with open(f_in) as f_in, open(f_out, 'w') as f_out:
        for line in f_in:
            line = line.split()
            line = list(map(lambda x: 'DIGIT' *len(x) if x.isdigit() else x, line))
            line = ' '.join(line)
            f_out.write('<s> ' + line +'\n')


def combine_intent_slot(f_slot, f_intent, f_combined):
    with open(f_slot) as f_slot, open(f_intent) as f_intent, open(f_combined, 'w') as f_combined:
        for slot, intent in zip(f_slot, f_intent):
            f_combined.write(intent[:-1] + ' ' + slot)


if __name__ == '__main__':
    # cnt_intent('train/label')
    # cnt_intent('train/uni_label')
    get_uni_intent('test/label', 'test/seq.in', 'test/seq.out', 'test/intent_seq.in','test/intent_seq.out')
    get_uni_intent('valid/label', 'valid/seq.in', 'valid/seq.out', 'valid/intent_seq.in','valid/intent_seq.out')
    get_uni_intent('train/label', 'train/seq.in', 'train/seq.out','train/intent_seq.in', 'train/intent_seq.out')
    # get_uni_intent('test/label', 'test/uni_label')
    # get_uni_intent('train/label', 'train/uni_label')
    # get_pad('test/seq.in', 'test/intent_seq.in')
    # get_pad('train/seq.in', 'train/intent_seq.in')
    # combine_intent_slot('test/seq.out', 'test/label', 'test/intent_seq.out')
    # get_uni_intent('train/label', 'train/seq.in', 'train/seq.out','train/intent_seq.in', 'train/intent_seq.out')
    # combine_intent_slot('train/seq.out', 'train/uni_label', 'train/intent_seq.out')
    # pass
    # get_cap_fea('train/seq_cap.in', 'train/seq_capfea.in')
    # capitalize_NNP_and_get_POS('test/seq.in', 'test/seq_cap.in','test/seq_pos.in')
    # capitalize_NNP_and_get_POS('train/seq.in', 'train/seq_cap.in', 'train/seq_pos.in')
    # get_cap_fea('train/seq_cap.in', 'train/seq_capfea.in')
    # get_cap_fea('test/seq_cap.in', 'test/seq_capfea.in')
    # del_intent('train/label','train/uni_label')
