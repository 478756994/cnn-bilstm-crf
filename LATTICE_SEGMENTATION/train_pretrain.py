import sys
from modules import pad, test, save_optimizer, load_optimizer, write_to_txt
import model 
import torch.nn as nn 
import torch 
from data import data_preprocess
from tqdm import tqdm

from param import Tensor, LongTensor, RNN_TYPE
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


CNN_OUTPUTS_SIZE = 32
KERNEL_SIZE_LIST = [1, 3, 5, 7]
NUM_DIRS = 2 #1单向2双向
NUM_LAYERS = 1
DROPOUT = 0.5
IMPOSSIBLE = -1e4
learning_rate = 5e-4
#实例化模型
train_x, train_y, seq_len = data.load_batch_data(data_class='train', batch_size=256)
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
if MODEL == 'Bi_GRU':
    mymodel = model.Bi_GRU(vocab_dict=data.vocab_dict, embedding_size=embedding_size, rnn_size=rnn_size, num_tags=5)

saved_models_path = '/home/hanshuo/Documents/Research/LATTICE_SEGMENTATION/saved_models_pretrain/'
mymodel.check_path = saved_models_path+MODEL+'_'+str(embedding_size)+'_'+str(rnn_size)+'_params.pkl'


#重载模型参数，这里只载入Bi-LSTM-CRF的参数
print('载入预训练模型Bi-LSTM-CRF')
pre_model = mymodel = model.Bi_LSTM_CRF(vocab_dict=data.vocab_dict, embedding_size=embedding_size+CNN_OUTPUTS_SIZE*len(KERNEL_SIZE_LIST), rnn_size=rnn_size, num_tags=5)
pre_model.restore_weights(pre_model.check_path)
pre_model_state = pre_model.state_dict()
#print(pre_model_state.keys())
del pre_model_state['global_step']
del pre_model_state['embed.char_embedding.weight']
model_dict = mymodel.state_dict()
model_dict.update(pre_model_state)
print('导入Bi-LSTM-CRF至CNN-Bi-LSTM-CRF')
mymodel.load_state_dict(model_dict)

#创建优化器
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)

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



#先对模型在验证集上测试
mymodel.eval()
scores, paths = mymodel(dev_x)
paths, _ = pad(paths, padding_id=data.class_to_id_fun(PAD))
paths = LongTensor(paths)
F_max = test(paths, dev_y, test_class_id_dict)
print('best model F_score on dev_data:{}'.format(str(F_max)))

F_on_test = 0    
#for epoch in tqdm(range(epochs)):
for epoch in range(epochs):
    #迭代batch，开始训练
    sum_loss = Tensor([0.])
    num_samples = 0 + 1e-2
    mymodel.train()
    for x, y in zip(train_x, train_y):
        num_samples += len(y)
        x = LongTensor(x)
        y = LongTensor(y)

        optimizer.zero_grad()
        loss = mymodel.loss(x, y)
        #print(loss)
        sum_loss += loss

        loss.backward()
        optimizer.step()
        mymodel.global_step += 1
    #print(num_samples)
    #在验证集上测试
    mymodel.eval()
    scores, paths = mymodel(dev_x)
    paths, _ = pad(paths, padding_id=data.class_to_id_fun(PAD))
    paths = LongTensor(paths)
    F = test(paths, dev_y, test_class_id_dict)
    if F>F_max:
        F_max = F
        mymodel.save_my_weights(mymodel.check_path)
        save_optimizer(optimizer, mymodel.check_path+'.optimizer')
        #在测试集上测试
        mymodel.eval()
        scores, paths = mymodel(test_x)
        paths, _ = pad(paths, padding_id=data.class_to_id_fun(PAD))
        paths = LongTensor(paths)
        F_on_test = test(paths, test_y, test_class_id_dict)
    print('best model F_score on dev_data:{}, best model F_score on test_data:{}'.format(str(F_max), str(F_on_test)))
    write_to_txt('best model F_score on dev_data:{}, best model F_score on test_data:{}'.format(str(F_max), str(F_on_test)), mymodel.check_path+'_log.txt')
    print('Episode {}:  gloable_step {}, loss {:.4f}'.format(
        epoch, mymodel.global_step[0], sum_loss[0]/num_samples))
#最终在测试集上测试
mymodel.restore_weights(mymodel.check_path)
load_optimizer(optimizer, mymodel.check_path+'.optimizer')

mymodel.eval()
scores, paths = mymodel(test_x)
paths, _ = pad(paths, padding_id=data.class_to_id_fun(PAD))
paths = LongTensor(paths)
F = test(paths, test_y, test_class_id_dict)

print('best F_score at:', F)    
write_to_txt('best F_score at:{:.6f}'.format(F), mymodel.check_path+'_log.txt')
