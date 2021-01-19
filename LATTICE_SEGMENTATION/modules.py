import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import collections
import numpy as np 
from param import *
from glove import Glove

#日志保存函数
def write_to_txt(text, filename):
    with open(filename, 'a+') as f:
        f.write(text+'\n')

#保存优化器
def save_optimizer(optimizer, save_path):
    '''
    Input:
        optimizer:一个torch优化器对象
            type:torch.optim
        save_path:目标位置
            type:string
    '''
    try:
        torch.save(optimizer.state_dict(), save_path)
        print('optimizer saved {}'.format(save_path))
    except Exception as e:
        print(e)
        print('fail to save')
#重载优化器
def load_optimizer(optimizer, save_path):
    '''
    Input:同save_optimizer
    '''
    try:
        optimizer.load_state_dict(torch.load(save_path))
        print('optimizer restored from: {}'.format(save_path))
    except Exception as e:
        print(e)
        print('no optimizer found...')

#Padding函数，将x中较小长度的序列以padding_id填充成其最大长度
def pad(x, padding_id):
    '''
    Padding函数，将x中较小长度的序列以padding_id填充成其最大长度
    Input:
        x:
        type:list
        e.g.:[[id0, id1...idn], [id0, id1...idm], [id0, id1...idc]...]由id序列组成的嵌套列表，子列表的长度不一
        padding_id：一个id，该id将被pad到x中
        type(padding_id):int
    return:
        padded_x:填充的x；type(padded_x):np.array
        seq_len:原x中每个序列的长度；type(seq_len):list
    '''
    seq_len = [len(seq) for seq in x]
    max_len = max(seq_len)
    padded_x = np.full((len(x), max_len), padding_id, np.int32)
            
    for j in range(len(x)):
        padded_x[j, :len(x[j])] = x[j]

    return padded_x, seq_len

def iv_test():
    pass 
    return 

def ovv_test():
    pass 
    return 

#计算指定类别的TP、FP、FN
def eva(y_hat, y, class_id):
    y_hat = y_hat.reshape([-1])
    y = y.reshape([-1])
    num_samples = y_hat.shape[0]

    ruler = LongTensor(np.ones(shape=[num_samples, ], dtype=np.int32) * class_id)

    TPandFP = ruler.eq(y_hat)
    TPandFN = ruler.eq(y)

    NonTP = torch.nonzero(torch.add(torch.abs(y_hat-ruler), torch.abs(y-ruler))).shape[0]

    nTP = y.shape[0] - int(NonTP)
    nTPandFP = torch.sum(TPandFP)
    nTPandFN = torch.sum(TPandFN)

    return nTP, int(nTPandFP), int(nTPandFN)

#测试函数
def test(y_hat, y, class_id_dict):
    #y_hat.shape:[batch_size, max_seq_len] 
    #type(y_hat):LongTensor
    #y: 同y_hat
    #class_to_id_fun：{'b':0, 'm':1, 'e':2, 's':3}
    #返回平均F-score
    #write_to_txt(text='******'*2+'test report'+'******'*2)
    F_ave = 0
    P_ave = 0
    R_ave = 0
    for class_tag, class_id in class_id_dict.items():
        TP, TPandFP, TPandFN = eva(y_hat, y, class_id)
        P = TP/(TPandFP+1e-3)
        R = TP/(TPandFN+1e-3)
        if TP>TPandFP:
            print('TP:{}, TPandFP:{}'.format(TP, TPandFP))
        if TP>TPandFN:
            print('TP:{}, TPandFN:{}'.format(TP, TPandFN))
        F = 2*P*R/(P+R+1e-3)
        F_ave += F
        P_ave += P 
        R_ave += R
        print('class {}: Precision {:.4f}, Recall {:.4f}, F1 {:.4f}'.format(class_tag, P, R, F))
        #write_to_txt(text='class {}: Precision {:.3f}, Recall {:.3f}, F1 {:.3f}'.format(class_tag, P, R, 2*P*R/(P+R+1e-3)))
    print('P-ave:{:.4f}, R-ave:{:.4f}'.format(P_ave/len(['b', 'm', 'e', 's']), R_ave/len(['b', 'm', 'e', 's'])))
    print('^^^^^^'*2+'test report'+'^^^^^^'*2)
    #write_to_txt(text='^^^^^^'*2+'test report'+'^^^^^^'*2)
    F_ave /= len(['b', 'm', 'e', 's'])
        
    return F_ave
#编码标签序列，切分字符序列

def seg_decoder(data, class_to_id_fun):
    '''
    Input:
        data:[character, tags]
            e.g.:[['北', '京',...], ['b', 'm',...]]
        class_to_id_fun:Data.class_to_id_fun
    '''

    x = data[0]
    y = data[1]
    segment = []
    #print(x, y)
    seg = ''
    for index, label in enumerate(y):
        flag = 0
        #如果flag为1,说明正在组合某个seg
        if label == class_to_id_fun('b'):
            if flag == 1:
                segment.append(seg)
            seg = str(x[index])
            flag=1
        if label == class_to_id_fun('m'):
            seg += str(x[index])
        if label == class_to_id_fun('e'):
            seg += str(x[index])
            flag = 0
            segment.append(seg)
            seg = ''
        if label == class_to_id_fun('s'):
            segment.append(str(x[index]))
            flag = 0
        
    return segment   

def decode_tags_into_index(tag_id_list, class_to_id_fun):
    segments = []
    seg = []
    for index, label in enumerate(tag_id_list):
        flag = 0
        #如果flag为1,说明正在组合某个seg
        if label == class_to_id_fun('b'):
            if flag == 1:
                segments.append(tuple(seg))
            seg.append(index)
            flag=1
        if label == class_to_id_fun('m'):
            seg.append(index)
        if label == class_to_id_fun('e'):
            seg.append(index)
            flag = 0
            segments.append(tuple(seg))
            seg = []
        if label == class_to_id_fun('s'):
            if flag == 1:
                segments.append(tuple(seg))
                seg = []
            segments.append(tuple([index, ]))
            flag = 0
        
    return segments   


def test_for_seg(model, data, x, y, PAD_IDX):
    '''
    Input:
        model:需要测试的模型实例
        x:字符数据
            type:torch.Tensor
            shape:[batch_size, max_seq_len]
        y：真实标签
            type:torch.Tensor
            shape:[batch_size, max_seq_len]
        data:数据实例
            type:data_preprocess.Data
    '''
    class_to_id_fun = data.class_to_id_fun
    seg_count = 0
    correct_count = 0
        
    y_list_temp = y.tolist()
    x_list_temp = x.tolist()
    #pred_list_temp = pred.tolist()
    _, pred_list = model(x)   
    y_list=[]
    x_list=[]
    #pred_list=[]
    for i, each_y in enumerate(y_list_temp):
        y_t = []
        x_t = []
        #pred_t = []
        for j, each_tag in enumerate(each_y):    
            if each_tag != PAD_IDX:
                x_t.append(x_list_temp[i][j])
                y_t.append(y_list_temp[i][j])
                #pred_t.append(pred_list_temp[i][j])
        y_list.append(y_t)
        x_list.append(x_t)
        #pred_list.append(pred_t)
    x_list = [[data.vocab_ivdict[c] for c in x] for x in x_list]
    c_index = []#标准切分样本的index列表    
    for i, y in enumerate(y_list):
        real_seg_index = decode_tags_into_index(y, class_to_id_fun)
        pred_seg_index = decode_tags_into_index(pred_list[i], class_to_id_fun)
        seg_count += len(real_seg_index)
        correct_count += len(set(real_seg_index)&set(pred_seg_index))
        c_index.append(real_seg_index)

    print('seg acc: {:.4f}'.format(correct_count/seg_count))
        
    return correct_count/seg_count, c_index

def test_for_oov(model, data, x, y, PAD_IDX, oov_set):
    '''
    Input:
        model:需要测试的模型实例
        x:字符数据
            type:torch.Tensor
            shape:[batch_size, max_seq_len]
        y：真实标签
            type:torch.Tensor
            shape:[batch_size, max_seq_len]
        data:数据实例
            type:data_preprocess.Data
    '''
    class_to_id_fun = data.class_to_id_fun
    seg_count = 0
    oov_count = 0
    correct_count = 0
    oov_correct_count = 0
        
    y_list_temp = y.tolist()
    x_list_temp = x.tolist()
    #pred_list_temp = pred.tolist()
    _, pred_list = model(x)   
    y_list=[]
    x_list=[]
    #pred_list=[]
    for i, each_y in enumerate(y_list_temp):
        y_t = []
        x_t = []
        #pred_t = []
        for j, each_tag in enumerate(each_y):    
            if each_tag != PAD_IDX:
                x_t.append(x_list_temp[i][j])
                y_t.append(y_list_temp[i][j])
                #pred_t.append(pred_list_temp[i][j])
        y_list.append(y_t)
        x_list.append(x_t)
        #pred_list.append(pred_t)
    x_list = [[data.vocab_ivdict[c] for c in x] for x in x_list]
    c_index = []#存在oov的样本且正确切分的index列表    
    for i, y in enumerate(y_list):
        chars = x_list[i]
        real_seg_index = decode_tags_into_index(y, class_to_id_fun)
        y_segment = seg_decoder(data=[chars, y], class_to_id_fun=class_to_id_fun)
        pred_seg_index = decode_tags_into_index(pred_list[i], class_to_id_fun)
        for j, seg in enumerate(y_segment):
            if seg in oov_set:
                oov_count += 1
                #print(oov_count)
                if real_seg_index[j] in pred_seg_index:
                    oov_correct_count += 1
                    c_index.append(i)
        seg_count += len(real_seg_index)
        correct_count += len(set(real_seg_index)&set(pred_seg_index))
        

    print('seg acc: {:.4f}'.format(correct_count/seg_count))
    print('oov or iv acc: {:.4f}'.format(oov_correct_count/oov_count))
    return correct_count/seg_count, oov_correct_count/oov_count, set(c_index)

#CRF中的函数
def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

class RNN(nn.Module):
    def __init__(self, embedding_size, rnn_size):
        super().__init__()
        self.batch_size = 0
        self.rnn_size = rnn_size
        # architecture
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = embedding_size,
            hidden_size = rnn_size // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        #self.out = nn.Linear(rnn_size) # RNN output to tag

    def init_state(self, b): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        h = self.rnn_size // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, input_x, mask):
        '''
        Input:
            input:
                type:Torch.Tensor
                shape:[batch_size, ]
        Return:
            h:
                type:Torch.Tensor
                shape:[batch_size, max_seq_len, embedding_size]
        '''
        hs = self.init_state(input_x.shape[0]) #获取数据的batch_size
        x = nn.utils.rnn.pack_padded_sequence(input_x, mask.sum(1).int().cpu(), enforce_sorted=False, batch_first = True)
        h, _ = self.rnn(x, hs)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        h *= mask.unsqueeze(2)
        return h

#位置编码
class PositionEmbed(nn.Module):
    '''
    Init:
        vocab_dict:char到Id的映射词典
            type:dict
        embed_size:自定义词向量维度
            type:Int
        max_seq_len:序列最大的长度
            type:Int
        use_abs:True为绝对位置，False为相对位置
            type:bool
    '''
    #vocab_dict是char到Id的映射词典，embed_size是自定义词向量维度,max_seq_len表示序列最大的长度
    #use_
    def __init__(self, vocab_dict, embed_size, use_abs=False, max_seq_len=34, pre_trained_path=None):
        super().__init__()
        #统计char词典中的char个数
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        #创建torch词向量函数
        if use_abs:
            self.PosEmbed = nn.Embedding(self.max_seq_len, self.embed_size)
        else:
            self.PosEmbed = self.get_sinusoid_encoding_table
        self.Embed = Embed(vocab_dict=vocab_dict, embed_size=embed_size, pre_trained_path=None)
        self = self.cuda() if use_gpu else self

    #获取词嵌入
    def forward(self, input_x):
        #input:
        #input_x.shape = [batch_size, seq_length]
        
        #return:
        #embeddings.shape = [batch_size, seq_length, embed_size]
        batch_size = input_x.shape[0]
        char_embedding = self.Embed(input_x)
        abs_position = [i for i in range(input_x.shape[-1])]
        abs_position = LongTensor([abs_position,]*batch_size)
        abs_position_embeddings = self.PosEmbed(abs_position)
        position_embedding = abs_position_embeddings+char_embedding
        return position_embedding

    def get_sinusoid_encoding_table(self, input_x):
        '''
        Input:
            shape:[batch_size, max_seq_len]
            type:Torch.LongTensor
        '''
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / self.embed_size)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(self.embed_size)]

        batch_size, n_position = input_x.shape
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return Tensor(sinusoid_table).unsqueeze(dim=0).repeat(batch_size, 1, 1)


#自定义词嵌入层，方便扩展功能
class Embed(nn.Module):
    #vocab_dict是char到Id的映射词典，embed_size是自定义词向量维度
    def __init__(self, vocab_dict, embed_size, pre_trained_path=None, pre_trained_model=None):
        super().__init__()
        #统计char词典中的char个数
        self.vocab_dict = vocab_dict
        self.vocab_size = len(vocab_dict.keys())
        self.embed_size = embed_size
        #创建torch词向量函数
        if pre_trained_path:
            self.char_embedding = self.load_pretrained(pre_trained_path, pre_trained_model)
        else:
            self.char_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self = self.cuda() if use_gpu else self

    #获取词嵌入
    def forward(self, input_x):
        #input:
        #input_x.shape = [batch_size, seq_length]
        
        #return:
        #embeddings.shape = [batch_size, seq_length, embed_size]
        embeddings = self.char_embedding(input_x)
        
        return embeddings

    def load_pretrained(self, pre_trained_path, pre_trained_model):
        weights = np.random.randn(self.vocab_size, self.embed_size)
        if pre_trained_model == 'glove':
            glove = Glove.load(pre_trained_path)
            for k, v in self.vocab_dict.items():
                if k != PAD:
                    weights[v,:] = glove.word_vectors[glove.dictionary[k]]
            print('Loaded GloVe pre training model ')
        elif pre_trained_model == 'word2vec':
            word2vec = gensim.models.Word2Vec.load(pre_trained_path)
            for k, v in self.vocab_dict.items():
                if k != PAD:
                    weights[v,:] = word2vec[k]
            print('Loaded Word2Vec pre training model ')
        pass 

        return nn.Embedding.from_pretrained(torch.FloatTensor(weights))

class CNN(nn.Module):
    def __init__(self, input_size, kernel_output_size, size_list=[3, 5, 7]):
        '''
        Input:
            input_size: embedding_size;Int
            output_size: 卷积核个数
            size_list: 所有卷积尺度
        '''
        super().__init__()
        self.input_size = input_size
        self.kernel_output_size = kernel_output_size
        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=self.input_size, 
            out_channels=kernel_output_size, kernel_size=size, padding=int((size-1)/2)).cuda() for size in size_list])
        self = self.cuda() if use_gpu else self

    def forward(self, input_x):
        '''
        Input:
            input_x:
                type:torch.Tensor
                shape:[batch_size, max_seq_len, embedding_size]
        Return:
            outputs:
                type:list
                sample:[conv1_output, conv2_output,...]
        '''
        outputs = []
        for conv1d in self.conv_list:
            output = conv1d(input_x.permute(0, 2, 1))
            outputs.append(output)
        pass 
        return outputs

class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.
    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores

class Four_Pos_Fusion_Embedding(nn.Module):
    def __init__(self,pe,four_pos_fusion,pe_ss,pe_se,pe_es,pe_ee,max_seq_len,hidden_size,mode):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.max_seq_len=max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.pe = pe
        self.four_pos_fusion = four_pos_fusion
        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        if self.four_pos_fusion == 'ff_linear':
            self.pos_fusion_forward = nn.Linear(self.hidden_size*4,self.hidden_size)

        elif self.four_pos_fusion == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4*self.hidden_size))

            # print('暂时不支持以gate融合pos信息')
            # exit(1208)
    def forward(self,pos_s,pos_e):
        batch = pos_s.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)

        if self.mode['debug']:
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))
            print('pos_ss:{}'.format(pos_ss))
            print('pos_se:{}'.format(pos_se))
            print('pos_es:{}'.format(pos_es))
            print('pos_ee:{}'.format(pos_ee))
        # B prepare relative position encoding
        max_seq_len = pos_s.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        pe_ss = self.pe_ss[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
        pe_se = self.pe_se[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_linear':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_two':
            pe_2 = torch.cat([pe_ss,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('2个位置合起来:{},{}'.format(pe_2.size(),size2MB(pe_2.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_2)
        elif self.four_pos_fusion == 'attn':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = self.w_r(pe_4.view(batch,max_seq_len,max_seq_len,4,self.hidden_size))
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion
            if self.mode['debug']:
                print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
                print(pe_4_fusion.size())

        elif self.four_pos_fusion == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_score = self.pos_gate_score(pe_4).view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
            gate_score = F.softmax(gate_score,dim=-2)
            pe_4_unflat = self.w_r(pe_4.view(batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion


        return rel_pos_embedding
