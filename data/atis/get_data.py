with open('train/seq.in') as f_in, open('train/seq.out') as f_out, open('train/uni_label') as f_label, open('train/train.txt', 'w') as f_result:
    for l_in, l_out, l_label in zip(f_in, f_out, f_label):
        l_in = l_in.split()
        l_out = l_out.split()
        for w_in, w_out in zip(l_in, l_out):
            f_result.write(w_in + ' ' + w_out+'\n' )
        f_result.write(l_label +'\n')

with open('valid/seq.in') as f_in, open('valid/seq.out') as f_out, open('valid/uni_label') as f_label, open('valid/valid.txt', 'w') as f_result:
    for l_in, l_out, l_label in zip(f_in, f_out, f_label):
        l_in = l_in.split()
        l_out = l_out.split()
        for w_in, w_out in zip(l_in, l_out):
            f_result.write(w_in + ' ' + w_out+'\n' )
        f_result.write(l_label +'\n')

with open('test/seq.in') as f_in, open('test/seq.out') as f_out, open('test/uni_label') as f_label, open('test/test.txt', 'w') as f_result:
    for l_in, l_out, l_label in zip(f_in, f_out, f_label):
        l_in = l_in.split()
        l_out = l_out.split()
        for w_in, w_out in zip(l_in, l_out):
            f_result.write(w_in + ' ' + w_out+'\n' )
        f_result.write(l_label +'\n')
