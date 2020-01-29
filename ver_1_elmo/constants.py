PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<UNK>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

SEED = 1
GPU = '-1'

# TAGS = 124 # for atis
# TAGS = 76 # for snips
TAGS = 20 # for snips test
ELMO_OPTIONS= "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
              "2x4096_512_2048cnn_2xhighway/" \
              "elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
              "2x4096_512_2048cnn_2xhighway/" \
              "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


