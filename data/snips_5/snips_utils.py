from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk
import stanfordnlp


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
    nlp = stanfordnlp.Pipeline(processors='tokenize,pos,lemma',tokenize_pretokenized=True)

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
            f_fea.write(fea+'\n')

if __name__ == '__main__':
    # get_cap_fea('train/seq_cap.in', 'train/seq_capfea.in')
    capitalize_NNP_and_get_POS('test/seq.in', 'test/seq_cap.in','test/seq_pos.in')
    capitalize_NNP_and_get_POS('train/seq.in', 'train/seq_cap.in', 'train/seq_pos.in')
    get_cap_fea('train/seq_cap.in', 'train/seq_capfea.in')
    get_cap_fea('test/seq_cap.in', 'test/seq_capfea.in')
    # del_intent('train/label','train/uni_label')
