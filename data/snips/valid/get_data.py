with open('seq.in') as f_in, open('seq.out') as f_out, open('label') as f_label, open('valid.txt', 'w') as f_result:
    for l_in, l_out, l_label in zip(f_in, f_out, f_label):
        l_in = l_in.split()
        l_out = l_out.split()
        for w_in, w_out in zip(l_in, l_out):
            f_result.write(w_in + ' ' + w_out+'\n' )
        f_result.write(l_label +'\n')
