import torch 
import numpy as np
import os
import collections
import re 

class BiGRU_CRF(torch.nn.Module):
    def __init__(self, embedding_dim=300, num_class=4, num_units=1000):
        super().__init__()
        
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.num_units = num_units
        self.dictionary_generator()
        max_words = len(self.dictionary.keys())
        
        #词嵌入层
        self.Embeddings = torch.nn.Embedding(max_words, embedding_dim)
        #双向GRU层
        self.bi_gru = torch.nn.LSTM(input_size=embedding_dim, hidden_size=num_units, 
                                   num_layers=1, batch_first=True,bidirectional=True)
        #全连接层
        self.fc = torch.nn.Linear(num_units*2, num_class+2)
        
        self.transitions = torch.nn.Parameter(torch.randn(num_class+2, num_class+2).to(device).requires_grad_())
        
        self.class_id = {'b': 0, 'm': 1, 'e': 2, 's': 3, '<s>': 4, '<e>': 5}
        self.class_to_id_fun = lambda x : self.class_id.get(x)
        
        #损失函数
        #self.Loss = torch.nn.CrossEntropyLoss(weight=None, size_average=False)
        self.global_step = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.int32), requires_grad=False)
        
        #优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print('load training-data...')
        self.train_x, self.train_y = self.load_data(*['train.src', 'train.trg'])
        print('load testing-data...')
        self.test_x, self.test_y = self.load_data(*['test.src', 'test.trg'])
        print('load dev-data...')
        self.dev_x, self.dev_y = self.load_data(*['dev.src', 'dev.trg'])
        print('*******************'*3)
        self.to(device)
        pass 
    
    
    def embedding_function(self, input):
        embeddings = self.Embeddings(input)
        return embeddings
    
    
    def _get_gru_features(self, input_x):
        embeddings = self.embedding_function(input_x)
        outputs, _ = self.bi_gru(embeddings)
        emission_score = self.fc(outputs)
        return emission_score.squeeze(0)
    
    def test_for_seg(self, x, y):
        seg_count = 0
        correct_count = 0
        for i, words in enumerate(x):
            _, path = model(words)
            
            real_seg = self.seg_decoder([words.squeeze().tolist(), y[i].tolist()])
            pred_seg = self.seg_decoder([words.squeeze().tolist(), path])
            seg_count += len(real_seg)
            for i, each_seg in enumerate(real_seg):
                try:
                    if each_seg == pred_seg[i]:
                        correct_count += 1  
                except:
                    break
        print('seg acc: {:.3f}'.format(correct_count/seg_count))
        
        return correct_count/seg_count
    
    def test(self, x, y):
        pred_path = torch.tensor([], device=device, dtype=torch.int32)
        real_path = torch.tensor([], device=device, dtype=torch.long)
        for i, words in enumerate(x):
            _, path = model(words)

            pred_path = torch.cat((pred_path, torch.tensor(path, device=device, dtype=torch.int32)), dim=-1)
            real_path = torch.cat((real_path, y[i]), dim=-1)
        
        print('******'*2+'test report'+'******'*2)

        write_to_txt(text='******'*2+'test report'+'******'*2)
        F_ave = 0
        for class_tag in ['b', 'm', 'e', 's']:
            TP, TPandFP, TPandFN = self.eva(pred_path, real_path, self.class_id[class_tag])
            P = TP/(TPandFP+1e-3)
            R = TP/(TPandFN+1e-3)
            if TP>TPandFP:
                print('TP:{}, TPandFP:{}'.format(TP, TPandFP))
            if TP>TPandFN:
                print('TP:{}, TPandFN:{}'.format(TP, TPandFN))
            F = 2*P*R/(P+R+1e-3)
            F_ave += F
            print('class {}: Precision {:.3f}, Recall {:.3f}, F1 {:.3f}'.format(class_tag, P, R, F))
            write_to_txt(text='class {}: Precision {:.3f}, Recall {:.3f}, F1 {:.3f}'.format(class_tag, P, R, 2*P*R/(P+R+1e-3)))
        print('^^^^^^'*2+'test report'+'^^^^^^'*2)
        write_to_txt(text='^^^^^^'*2+'test report'+'^^^^^^'*2)
        F_ave /= len(['b', 'm', 'e', 's'])
        
        return F_ave
    
    
    def seg_decoder(self, data):
        # data.shape = [character, tags]
        # seg = []
        x = data[0]
        y = data[1]
        segment = []
        #print(x, y)
        for index, label in enumerate(y):
            seg = ''
            if label == self.class_to_id_fun('b'):
                seg = str(x[index])
            if label == self.class_to_id_fun('m'):
                seg += str(x[index])
            if label == self.class_to_id_fun('e'):
                seg += str(x[index])
                segment.append(seg)
                del seg
            if label == self.class_to_id_fun('s'):
                segment.append(str(x[index]))
        
        return segment
    
    def load_data(self, file_x, file_y):
        with open(file_x) as f:
            x_lines = [re.sub('\ufeff', '', line.strip('\n')).split(' ') for line in f.readlines()]
            x_lines = [[self.word_to_id_fun(word) for word in line] for line in x_lines]
        
        with open(file_y) as f:
            y_lines = [re.sub('\ufeff', '', line.strip('\n')).split(' ') for line in f.readlines()]
            y_lines = [[self.class_to_id_fun(tag) for tag in line] for line in y_lines]
        x = []
        y = []
        for index, line in enumerate(y_lines):
            if line != [None]:
                x.append(torch.unsqueeze(torch.tensor(x_lines[index], dtype=torch.long).cuda(), dim=0))
                y.append(torch.tensor(y_lines[index], dtype=torch.long).cuda())
        return [x, y]
   
    
    def dictionary_generator(self, path='./'):
        all_src = [] 
        for file_name in os.listdir(path):
            if file_name.split('.')[-1] == 'src':
                all_src.append(file_name)
        all_words = []
        for file_name in all_src:
            with open(file_name, 'r',encoding='utf-8-sig') as f:
                all_texts = [re.sub(r' |\ufeff', '', line.strip('\n')) for line in f.readlines()]
                for each_text in all_texts:
                    all_words.extend(each_text)

        word_count = collections.Counter(all_words).most_common(100)
        words_set, _ = zip(*word_count)
        words_set += (' ', )   
        self.dictionary = dict(zip(words_set, range(len(words_set))))
        self.word_to_id_fun = lambda x : self.dictionary.get(x, len(words_set) - 2)
       
    
    
    def save_my_weights(self, path=check_path):
        try:
            torch.save(self.state_dict(), path)
            print('weights saved: {}'.format(path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, path=check_path):
        try:
            model.load_state_dict(torch.load(path))
            print('weights restored from {}'.format(path))
        except:
            print('model weights not found...')
            
      
    
    def eva(self, pred, y, class_id):
        y_ = pred
        num_samples = y_.shape[0]

        ruler = torch.ones(size=[num_samples, ], device=device, dtype=torch.int32) * class_id

        TPandFP = ruler.eq(y_)
        TPandFN = ruler.eq(y)

        NonTP = torch.nonzero(torch.add(torch.abs(y_-ruler), torch.abs(y-ruler))).shape[0]

        nTP = y.shape[0] - int(NonTP)
        nTPandFP = torch.sum(TPandFP)
        nTPandFN = torch.sum(TPandFN)

        return nTP, int(nTPandFP), int(nTPandFN)
    
    def neg_log_likelihood(self, words, tags):  # 求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        frames = self._get_gru_features(words)  # emission score at each frame
        gold_score = self._score_sentence(frames, tags)  # 正确路径的分数
        forward_score = self._forward_alg(frames)  # 所有路径的分数和
        # -(正确路径的分数 - 所有路径的分数和）;注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
        return forward_score - gold_score, frames

    def _get_lstm_features(self, words):  # 求出每一帧对应的隐向量
        # LSTM输入形状(seq_len, batch=1, input_size); 教学演示 batch size 为1
        embeds = self.embedding_function(words).cuda()
        # 随机初始化LSTM的隐状态H
        hidden = torch.randn(2, 1, self.hidden_dim // 2, device=device), torch.randn(2, 1, self.hidden_dim // 2, device=device)
        lstm_out, _hidden = self.lstm(embeds, hidden)
        return self.hidden2tag(lstm_out.squeeze(0))  # 把LSTM输出的隐状态张量去掉batch维，然后降维到tag空间

    def _score_sentence(self, frames, tags):
        """
        求路径pair: frames->tags 的分值
        index:      0   1   2   3   4   5   6
        frames:     F0  F1  F2  F3  F4
        tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>
        """
        tags_tensor = torch.cat((torch.tensor([1 * self.class_id[START_TAG]], device=device), tags), dim=-1)  # 注意不要+[END_TAG]; 结尾有处理
        score = torch.zeros(1,device=device)
        for i, frame in enumerate(frames):  # 沿途累加每一帧的转移和发射
            score += self.transitions[tags_tensor[i], tags_tensor[i + 1]] + frame[tags_tensor[i + 1]]
        return score + self.transitions[tags_tensor[-1], self.class_id[END_TAG]]  # 加上到END_TAG的转移

    def _forward_alg(self, frames):
        """ 给定每一帧的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母 """
        alpha = torch.full((1, self.num_class+2), -10000.0, device=device)
        alpha[0][self.class_id[START_TAG]] = 0  # 初始化分值分布. START_TAG是log(1)=0, 其他都是很小的值 "-10000"
        for frame in frames:
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            alpha = log_sum_exp(alpha.T + frame.unsqueeze(0) + self.transitions)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(alpha.T + 0 + self.transitions[:, [self.class_id[END_TAG]]]).flatten()

    def _viterbi_decode(self, frames):
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.num_class+2), -10000., device=device)
        alpha[0][self.class_id[START_TAG]] = 0
        for frame in frames:
            # 这里跟 _forward_alg()稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = alpha.T + frame.unsqueeze(0) + self.transitions
            backtrace.append(smat.argmax(0))  # 当前帧每个状态的最优"来源"
            alpha = log_sum_exp(smat)  # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径

        # 回溯路径
        smat = alpha.T + 0 + self.transitions[:, [self.class_id[END_TAG]]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return log_sum_exp(smat).item(), best_path[::-1]  # 返回最优路径分值 和 最优路径

    def forward(self, words):  # 模型inference逻辑
        lstm_feats = self._get_gru_features(words)  # 求出每一帧的发射矩阵
        #print(lstm_feats)
        return self._viterbi_decode(lstm_feats)  # 采用已经训好的CRF层,