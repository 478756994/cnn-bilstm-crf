from modules import Embed, CNN, RNN, CRF, PositionEmbed
import torch.nn as nn 
import torch 
from data import data_preprocess
'''
from param import use_gpu
from param import PAD, SOS, SOS_IDX, EOS, EOS_IDX, UNK
from param import randn, zeros
'''
from param import *
data = data_preprocess.Data()
PAD_IDX = data.class_to_id_fun(PAD)
UNK_IDX = data.word_to_id_fun(PAD)

'''
PAD = "<PAD>" # padding
SOS, SOS_IDX = "<SOS>", 1 # start of sequence
EOS, EOS_IDX = "<EOS>", 2 # end of sequence
UNK = "<UNK>" # unknown token
RNN_TYPE = 'LSTM'
NUM_DIRS = 2
NUM_LAYERS = 1
DROPOUT = 0.5
IMPOSSIBLE = -1e4
'''

class Bi_LSTM_CNN_CRF(nn.Module):
    def __init__(self, vocab_dict, kernel_output_size, kernel_size_list, 
    embedding_size, rnn_size, num_tags):
        super().__init__()
        self.check_path = saved_models_path+'Bi_LSTM_CNN_CRF_'+str(embedding_size)+'_'+str(rnn_size)+'_params.pkl'
        self.global_step = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.embed = Embed(vocab_dict, embedding_size, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        #self.embed = PositionEmbed(max_seq_len=34, vocab_dict=vocab_dict, embed_size=embedding_size, use_abs=USE_ABS, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        self.cnn = CNN(input_size=embedding_size, kernel_output_size=kernel_output_size, size_list=kernel_size_list)
        self.rnn = RNN(rnn_size=rnn_size, embedding_size=embedding_size)
        self.crf = CRF(in_features=rnn_size+kernel_output_size*len(kernel_size_list), num_tags=num_tags)
        self = self.cuda() if use_gpu else self
        
    def forward(self, input_x):
        #传入的input_x必须是torch.LongTensor
        #input_x:[batch_size, max_seq_len]
        self.zero_grad()
        #self.rnn.batch_size = input_y.size(0)
        #self.crf.batch_size = input_y.size(0)
        
        #？？？UNK_IDX表示X中的占位符,后续需要为X单独添加占位符。因为未传入input_y，此处使用input_x计算mask。
        mask = input_x[:, :].le(UNK_IDX-1).float()
        x_embed = self.embed(input_x)
        cnn_embed = self.cnn(x_embed)
        #拼接Embedding
        cnn_embed = torch.cat(cnn_embed, dim=1).permute(0, 2, 1) 
        features = self.rnn(x_embed, mask)
        features = torch.cat([features, cnn_embed], dim=-1)
        scores, paths = self.crf(features, mask)

        return scores, paths

    def loss(self, input_x, input_y):
        #传入的input_x、input_y必须是torch.LongTensor
        self.zero_grad()
        mask = input_y[:, :].le(PAD_IDX-1).float() 
        x_embed = self.embed(input_x)
        cnn_embed = self.cnn(x_embed)
        #拼接Embedding
        cnn_embed = torch.cat(cnn_embed, dim=1).permute(0, 2, 1) 
        features = self.rnn(x_embed, mask)
        features = torch.cat([features, cnn_embed], dim=-1)
        loss = self.crf.loss(features, input_y, mask)

        return loss

    def save_my_weights(self, model_path):
        try:
            torch.save(self.state_dict(), model_path)
            print('weights saved: {}'.format(model_path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path))
            print('weights restored from {}'.format(model_path))
        except Exception as e:
            print(e)
            print('model weights not found...')

class CNN_for_Seq2Seq(nn.Module):
    def __init__(self, vocab_dict, kernel_output_size, kernel_size_list, 
    embedding_size, num_tags):
        super().__init__()
        self.check_path = saved_models_path+'CNN_for_Seq2Seq'+str(embedding_size)+'_'+str(kernel_output_size)+'_params.pkl'
        self.global_step = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.embed = Embed(vocab_dict, embedding_size, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        #self.embed = PositionEmbed(max_seq_len=34, vocab_dict=vocab_dict, embed_size=embedding_size, use_abs=USE_ABS, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        self.cnn = CNN(input_size=embedding_size, kernel_output_size=kernel_output_size, size_list=kernel_size_list)
        self.crf = CRF(in_features=kernel_output_size*len(kernel_size_list), num_tags=num_tags)
        self = self.cuda() if use_gpu else self
        
    def forward(self, input_x):
        #传入的input_x必须是torch.LongTensor
        #input_x:[batch_size, max_seq_len]
        self.zero_grad()
        #self.rnn.batch_size = input_y.size(0)
        #self.crf.batch_size = input_y.size(0)
        
        #？？？UNK_IDX表示X中的占位符,后续需要为X单独添加占位符。因为未传入input_y，此处使用input_x计算mask。
        mask = input_x[:, :].le(UNK_IDX-1).float()
        x_embed = self.embed(input_x)
        cnn_embed = self.cnn(x_embed)
        cnn_embed = torch.cat(cnn_embed, dim=1).permute(0, 2, 1) #[batch_size, max_seq_len, kernel_output_size*len(kernel_size_list)]
        scores, paths = self.crf(cnn_embed, mask)

        return scores, paths

    def loss(self, input_x, input_y):
        #传入的input_x、input_y必须是torch.LongTensor
        self.zero_grad()
        mask = input_y[:, :].le(PAD_IDX-1).float() 
        x_embed = self.embed(input_x)
        cnn_embed = self.cnn(x_embed)
        cnn_embed = torch.cat(cnn_embed, dim=1).permute(0, 2, 1) #[batch_size, max_seq_len, kernel_output_size*len(kernel_size_list)]
        loss = self.crf.loss(cnn_embed, input_y, mask)

        return loss

    def save_my_weights(self, model_path):
        try:
            torch.save(self.state_dict(), model_path)
            print('weights saved: {}'.format(model_path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path))
            print('weights restored from {}'.format(model_path))
        except Exception as e:
            print(e)
            print('model weights not found...')

class CNN_Bi_LSTM_CRF(nn.Module):
    def __init__(self, vocab_dict, kernel_output_size, kernel_size_list, 
    embedding_size, rnn_size, num_tags):
        super().__init__()
        self.check_path = saved_models_path+'CNN_Bi_LSTM_CRF_'+str(embedding_size)+'_'+str(rnn_size)+'_params.pkl'
        self.global_step = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.embed = Embed(vocab_dict, embedding_size, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        #self.embed = PositionEmbed(max_seq_len=34, vocab_dict=vocab_dict, embed_size=embedding_size, use_abs=USE_ABS, , pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        self.cnn = CNN(input_size=embedding_size, kernel_output_size=kernel_output_size, size_list=kernel_size_list)
        self.rnn = RNN(rnn_size=rnn_size, embedding_size=embedding_size+kernel_output_size*len(kernel_size_list))
        self.crf = CRF(in_features=rnn_size, num_tags=num_tags)
        self = self.cuda() if use_gpu else self
        
    def forward(self, input_x):
        #传入的input_x必须是torch.LongTensor
        #input_x:[batch_size, max_seq_len]
        self.zero_grad()
        #self.rnn.batch_size = input_y.size(0)
        #self.crf.batch_size = input_y.size(0)
        
        #？？？UNK_IDX表示X中的占位符,后续需要为X单独添加占位符。因为未传入input_y，此处使用input_x计算mask。
        mask = input_x[:, :].le(UNK_IDX-1).float()
        x_embed = self.embed(input_x)
        cnn_embed = self.cnn(x_embed)
        #拼接Embedding
        cnn_embed = torch.cat(cnn_embed, dim=1).permute(0, 2, 1) #[batch_size, cnn_output_size, max_seq_len]
        x_embed = torch.cat([x_embed, cnn_embed], dim=-1)
        features = self.rnn(x_embed, mask)
        scores, paths = self.crf(features, mask)

        return scores, paths

    def loss(self, input_x, input_y):
        #传入的input_x、input_y必须是torch.LongTensor
        self.zero_grad()
        mask = input_y[:, :].le(PAD_IDX-1).float() 
        x_embed = self.embed(input_x)
        cnn_embed = self.cnn(x_embed)
        #拼接Embedding
        cnn_embed = torch.cat(cnn_embed, dim=1).permute(0, 2, 1) #[batch_size, cnn_output_size, max_seq_len]
        x_embed = torch.cat([x_embed, cnn_embed], dim=-1)
        features = self.rnn(x_embed, mask)

        loss = self.crf.loss(features, input_y, mask)

        return loss

    def save_my_weights(self, model_path):
        try:
            torch.save(self.state_dict(), model_path)
            print('weights saved: {}'.format(model_path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path))
            print('weights restored from {}'.format(model_path))
        except Exception as e:
            print(e)
            print('model weights not found...')

class Bi_LSTM_CRF(nn.Module):
    def __init__(self, vocab_dict, embedding_size, rnn_size, num_tags):
        super().__init__()
        self.check_path = saved_models_path+'Bi_LSTM_CRF_'+str(embedding_size)+'_'+str(rnn_size)+'_params.pkl'
        self.global_step = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.embed = Embed(vocab_dict, embedding_size, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        self.rnn = RNN(rnn_size=rnn_size, embedding_size=embedding_size)
        self.crf = CRF(in_features=rnn_size, num_tags=num_tags)
        self = self.cuda() if use_gpu else self
        
    def forward(self, input_x):
        #传入的input_x必须是torch.LongTensor
        #input_x:[batch_size, max_seq_len]
        self.zero_grad()
        #self.rnn.batch_size = input_y.size(0)
        #self.crf.batch_size = input_y.size(0)
        
        #？？？UNK_IDX表示X中的占位符,后续需要为X单独添加占位符。因为未传入input_y，此处使用input_x计算mask。
        mask = input_x[:, :].le(UNK_IDX-1).float()
        x_embed = self.embed(input_x)
        features = self.rnn(x_embed, mask)
        scores, paths = self.crf(features, mask)

        return scores, paths

    def loss(self, input_x, input_y):
        #传入的input_x、input_y必须是torch.LongTensor
        self.zero_grad()
        mask = input_y[:, :].le(PAD_IDX-1).float() 
        x_embed = self.embed(input_x)
        features = self.rnn(x_embed, mask)

        loss = self.crf.loss(features, input_y, mask)

        return loss

    def save_my_weights(self, model_path):
        try:
            torch.save(self.state_dict(), model_path)
            print('weights saved: {}'.format(model_path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path))
            print('weights restored from {}'.format(model_path))
        except Exception as e:
            print(e)
            print('model weights not found...')
          
class Bi_GRU(nn.Module):
    def __init__(self, vocab_dict, embedding_size, rnn_size, num_tags):
        super().__init__()
        self.check_path = saved_models_path+'Bi_GRU_'+str(embedding_size)+'_'+str(rnn_size)+'_params.pkl'
        self.global_step = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.embed = Embed(vocab_dict, embedding_size, pre_trained_model=pre_trained_model, pre_trained_path=pre_trained_path)
        self.num_tags = num_tags
        self.rnn = RNN(rnn_size=rnn_size, embedding_size=embedding_size)
        self.fc = nn.Linear(in_features=rnn_size, out_features=num_tags+1)
        self.Loss = torch.nn.CrossEntropyLoss(weight=None, size_average=True)
        self = self.cuda() if use_gpu else self
        
    def forward(self, input_x):
        #传入的input_x必须是torch.LongTensor
        #input_x:[batch_size, max_seq_len]
        self.zero_grad()
        #self.rnn.batch_size = input_y.size(0)
        #self.crf.batch_size = input_y.size(0)
        
        #？？？UNK_IDX表示X中的占位符,后续需要为X单独添加占位符。因为未传入input_y，此处使用input_x计算mask。
        mask = input_x[:, :].le(UNK_IDX-1).float()
        x_embed = self.embed(input_x)
        features = self.rnn(x_embed, mask)
        logits = self.fc(features)
        pred = torch.argmax(logits, dim=-1)
        paths = [pred[i][:mask[i].sum().int()].tolist() for i in range(pred.shape[0])]

        return logits, paths

    def loss(self, input_x, input_y):
        #logits.shape == [batch_size, steps, num_class]
        #input_y.shape == [batch_size, steps]
        self.zero_grad()
        mask = input_y[:, :].le(PAD_IDX-1).float() 
        logits, _ = self.forward(input_x)
        logits = logits.reshape([-1, self.num_tags+1])
        input_y = input_y.reshape([-1])
        #print(logits, input_y)
        index = torch.nonzero(mask.reshape([-1])).reshape([-1])
        logits = logits.index_select(0, index)
        #print(logits.shape)
        input_y = input_y.index_select(0, index)
        #print(input_y.shape)
        loss = self.Loss(logits, input_y)
        return loss

    def save_my_weights(self, model_path):
        try:
            torch.save(self.state_dict(), model_path)
            print('weights saved: {}'.format(model_path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, model_path):
        try:
            self.load_state_dict(torch.load(model_path))
            print('weights restored from {}'.format(model_path))
        except Exception as e:
            print(e)
            print('model weights not found...')



