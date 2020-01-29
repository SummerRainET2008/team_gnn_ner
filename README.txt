ver_1_Elmo:
    To run the code and reproduce the results:
    1. run python preprocess.py
    2. run python train_SLSTM.py


ver_2_roberta:
    Change datasets to test:
    1. in make_features.py:
       process(
           "data/atis/train/intent_seq.in", "data/atis/train/intent_seq.out",
           f"{param.path_feat}/train.pydict", param
         )
         process(
           "data/atis/train/intent_seq.in", "data/atis/train/intent_seq.out",
           f"{param.path_feat}/vali.pydict", param
         )

    2. in param.py
       self.file_for_label_tokenizer = 'data/atis/train/intent_seq.out'

    3. in __init__.py
       TAGS, INTENTS


    Run:
    1. python ver_1_roberta/make_features.py
    2. python ver_1_roberta/train.py
