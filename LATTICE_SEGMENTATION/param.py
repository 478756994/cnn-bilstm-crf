import torch 
#定义数据文件路径
data_path = '/home/hanshuo/Documents/Research/DATA/Uyghur/Segmentation/'
saved_models_path = '/home/hanshuo/Documents/Research/LATTICE_SEGMENTATION/saved_models/'
use_gpu = torch.cuda.is_available()
#使用预训练词向量类型与路径
pre_trained_model = None #'glove'  #glove/word2vec/None
pre_trained_path = None #'/home/hanshuo/Documents/Research/LATTICE_SEGMENTATION/data/glove_models/5_256.glove'
#使用位置编码True:绝对编码，FALSE：相对编码
USE_ABS = False



embedding_size = 256
rnn_size = 2048

PAD = "<PAD>" # padding
SOS, SOS_IDX = "<SOS>", 1 # start of sequence
EOS, EOS_IDX = "<EOS>", 2 # end of sequence
UNK = "<UNK>" # unknown token

RNN_TYPE = 'LSTM'
NUM_DIRS = 2
NUM_LAYERS = 1
DROPOUT = 0.5
IMPOSSIBLE = -1e4


Tensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
randn = lambda *x: torch.randn(*x).cuda() if use_gpu else torch.randn
zeros = lambda *x: torch.zeros(*x).cuda() if use_gpu else torch.zeros
