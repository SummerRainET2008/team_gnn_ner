import transformer.Constants as Constants
with open('intent_seq.out', 'w') as new_seq:
    with open('seq.out', 'r') as seq_file, open('label', 'r') as label_file:
        for seq_line, label in zip(seq_file, label_file):
            new_line = label[:-1]+ ' ' + seq_line
            new_seq.write(new_line)

with open('intent_seq.in','w') as new_seq:
    with open('seq.in','r') as seq:
        for line in seq:
            new_seq.write(Constants.BOS_WORD+' ' +line)
