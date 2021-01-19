import sys
from modules import pad, test, save_optimizer, load_optimizer, write_to_txt, test_for_seg, test_for_oov
import model 
import torch.nn as nn 
import torch 
from data import data_preprocess
from tqdm import tqdm

from param import Tensor, LongTensor
from param import use_gpu
from param import PAD, SOS, SOS_IDX, EOS, EOS_IDX, UNK

MODEL = sys.argv[1]
'''
#网络参数
#MODEL = 'Bi_LSTM_CRF'
#MODEL = 'CNN_Bi_LSTM_CRF'
#MODEL = 'CNN_for_Seq2Seq'
#MODEL = 'Bi_LSTM_CNN_CRF'
'''

epochs = 1050
batch_size = 512
embedding_size = 256
rnn_size = 2048

#实例化Data载入数据
data = data_preprocess.Data()
PAD_IDX = data.class_to_id_fun(PAD)
UNK_IDX = data.word_to_id_fun(PAD)


RNN_TYPE = 'LSTM'
CNN_OUTPUTS_SIZE = 32
KERNEL_SIZE_LIST = [1, 3, 5, 7]
NUM_DIRS = 2 #1单向2双向
NUM_LAYERS = 1
DROPOUT = 0.5
IMPOSSIBLE = -1e4
learning_rate = 1e-4
#实例化模型，试图重载模型
if MODEL == 'Bi_LSTM_CRF':
    mymodel = model.Bi_LSTM_CRF(vocab_dict=data.vocab_dict, embedding_size=embedding_size, rnn_size=rnn_size, num_tags=5)
elif MODEL == 'CNN_Bi_LSTM_CRF':
    mymodel = model.CNN_Bi_LSTM_CRF(vocab_dict=data.vocab_dict, 
    kernel_output_size=CNN_OUTPUTS_SIZE, kernel_size_list=KERNEL_SIZE_LIST, 
    embedding_size=embedding_size, rnn_size=rnn_size, num_tags=5)
elif MODEL == 'CNN_for_Seq2Seq':
    mymodel = model.CNN_for_Seq2Seq(vocab_dict=data.vocab_dict, kernel_output_size=CNN_OUTPUTS_SIZE, kernel_size_list=KERNEL_SIZE_LIST, 
    embedding_size=embedding_size, num_tags=5)
elif MODEL == 'Bi_LSTM_CNN_CRF':
    mymodel = model.Bi_LSTM_CNN_CRF(vocab_dict=data.vocab_dict, 
    kernel_output_size=CNN_OUTPUTS_SIZE, kernel_size_list=KERNEL_SIZE_LIST, 
    embedding_size=embedding_size, rnn_size=rnn_size, num_tags=5)
elif MODEL == 'Bi_GRU':
    mymodel = model.Bi_GRU(vocab_dict=data.vocab_dict, embedding_size=embedding_size, rnn_size=rnn_size, num_tags=5)

#创建优化器
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)
mymodel.restore_weights(mymodel.check_path)
load_optimizer(optimizer, mymodel.check_path+'.optimizer')

#准备测试的tags/class
test_class_id_dict = data.class_id.copy()
del test_class_id_dict[PAD]
#载入验证集数据，以备测试
dev_x, dev_y, _ = data.load_batch_data(data_class='dev')
dev_x = LongTensor(dev_x)
dev_y = LongTensor(dev_y)
#载入测试数据集，以备测试
test_x, test_y, _ = data.load_batch_data(data_class='test')
test_x = LongTensor(test_x)
test_y = LongTensor(test_y)

dev_iv, dev_oov = data.get_iv_oov('dev')
test_iv, test_oov = data.get_iv_oov('test')
#进行测试
print('***********testing MODEL{}...***********'.format(MODEL))
mymodel.eval()

scores, paths = mymodel(dev_x)
paths, _ = pad(paths, padding_id=data.class_to_id_fun(PAD))
paths = LongTensor(paths)
F_max = test(paths, dev_y, test_class_id_dict)

print('{}: best model F_score on dev_data:{}'.format(MODEL, str(F_max)))
#_, _ = test_for_seg(model=mymodel, data=data, x=dev_x, y=dev_y, PAD_IDX=PAD_IDX)
_, _, _ = test_for_oov(model=mymodel, data=data, x=dev_x, y=dev_y, PAD_IDX=PAD_IDX, oov_set=dev_oov)
mymodel.eval()
scores, paths = mymodel(test_x)
paths, _ = pad(paths, padding_id=data.class_to_id_fun(PAD))
paths = LongTensor(paths)
F = test(paths, test_y, test_class_id_dict)

print('{}: best model F_score on test_data:{}'.format(MODEL, str(F)))    
#_, _ = test_for_seg(model=mymodel, data=data, x=test_x, y=test_y, PAD_IDX=PAD_IDX)
_, _, _ = test_for_oov(model=mymodel, data=data, x=test_x, y=test_y, PAD_IDX=PAD_IDX, oov_set=test_oov)



