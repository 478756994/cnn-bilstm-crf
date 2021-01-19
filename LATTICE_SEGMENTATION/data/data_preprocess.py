import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from gensim.models import word2vec
from glove import Glove
from glove import Corpus
import os
import re
from tqdm import tqdm
import torch 
import sys 
sys.path.append("..")
import modules
from param import data_path
'''
以下命令安装glove_python, 避免编译错误
cd /tmp \
&& curl -o glove_python.zip -OL https://github.com/maciejkula/glove-python/archive/master.zip \
&& unzip glove_python.zip \
&& rm -f glove_python.zip \
&& cd glove-python-master \
&& cythonize -X language_level=3 -f -i glove/corpus_cython.pyx \
&& cythonize -X language_level=3 -f -i glove/glove_cython.pyx \
&& cythonize -X language_level=3 -f -i glove/metrics/accuracy_cython.pyx \
&& pip install . \
&& cd .. \
&& rm -rf glove-python-master
'''

PAD = "<PAD>" # padding
SOS, SOS_IDX = "<SOS>", 1 # start of sequence
EOS, EOS_IDX = "<EOS>", 2 # end of sequence
#UNK = "<UNK>" # unknown token
tag_set = ['b', 'm', 'e', 's']

def load_file(filename):
    with open(filename) as f:
        data = f.readlines()
        pass
    pass
    return [i.strip('\n').split(' ') for i in data]

#根据tags中的bmes信息，将character还原为切分
def decoder(data):
    # data.shape = [character, tags]
    # return:[seg0, seg1...]
    x = data[0]
    y = data[1]
    segment = []
    for index, label in enumerate(y):
        if label == 'b':
            seg = x[index]
        if label == 'm':
            seg += x[index]
        if label == 'e':
            seg += x[index]
            segment.append(seg)
            del seg
        if label == 's':
            segment.append(x[index])

    pass

    return segment


def count(all_text):
    # all_text must be a list which consists all characters
    c = Counter(all_text[0])
    pass
    return c


def main(group):
    #group=['train', 'test', 'dev']
    #返回：某个group数据的词汇Counter
    train_x = load_file(group + '.src')
    train_y = load_file(group + '.trg')
    segments = []
    for i in range(len(train_y)):
        segments.extend(decoder([train_x[i], train_y[i]]))
    pass
    print('all segments: ', len(segments))
    c = Counter(segments)

    count_1 = []
    count_2 = []
    count_3 = []
    count_4 = []
    count_other = []
    for key, value in c.items():
        if value == 1:
            count_1.append(key)
        elif value == 2:
            count_2.append(key)
        elif value == 3:
            count_3.append(key)
        elif value == 4:
            count_4.append(key)
        else:
            count_other.append(key)
    size = [len(count_1), len(count_2), len(count_3), len(count_4), len(count_other)]
    labels = ['counts:1', 'counts:2', 'counts:3', 'counts:4', 'counts:>=5']
    plt.pie(x=size, labels=labels, autopct='%1.1f%%')
    # plt.show()
    plt.savefig(group + '_seg_count.png')
    plt.close('all')
    return c

#为了方便理解维语字母，创建一个英文字母映射表
vocab = {'غ': 'a',
         'ا': 'b',
         'ۆ': 'c',
         'ن': 'd',
         'ڭ': 'e',
         'ۈ': 'f',
         'ب': 'g',
         'ە': 'h',
         'ى': 'i',
         'ك': 'j',
         'ل': 'k',
         'ۇ': 'l',
         'ق': 'm',
         'پ': 'n',
         'ئ': 'o',
         'ۋ': 'p',
         'ي': 'q',
         'ھ': 'r',
         'ې': 's',
         'ژ': 't',
         'ز': 'u',
         'ر': 'v',
         'ت': 'w',
         'س': 'x',
         'و': 'y',
         'خ': 'z',
         'ج': r'#',
         'ف': r'!',
         'چ': r'@',
         'د': r'$',
         'گ': r'%',
         'م': r'*',
         'ش': r'^'}

#从seg格式文件中载入数据的类，功能：数据读取、清洗、分割
class DataLoader():
    #初始化时，指定包含所有seg文件的路径
    #初始化后，需要获取数据中所有的切分：调用self.get_seg()
    #初始化后，需要获取数据中所有的词汇：调用self.get_words()
    def __init__(self, path='/home/hanshuo/Documents/Research/THUUyMorph/seg'):
        self.path = path
        self.files_list = self.find_file(path=self.path)
        pass


    #在path路径中找到所有以file_extension作为后缀的文件
    def find_file(self, path, file_extension='.seg'):
        #path:'./THUUyMorph/seg'
        #file_extension='.seg'
        #return:[file_name_0, file_name_1...]
        file_list = os.listdir(path)
        doc_list = []
        for file_name in file_list:
            if os.path.splitext(file_name)[-1] == file_extension:
                doc_list.append(file_name)

        return file_list


    #读取文件名为filename的文件
    def read_file(self, filename):
        #filename='xxx.seg'，文件名不包括路径前缀
        #results:[[str1, str2...], [str1...], ...]
        
        #使用正则表达式匹配标签内的内容
        with open(self.path + '/' + filename) as f:
            text = f.readlines()
        text = ''.join(text)
        text = re.sub('\n', '', text)
        text = re.sub('\'', '', text)
        patt_0 = re.compile(r'<S ID=\d+>(.*?)</S>')
        results = patt_0.findall(text)
        results = [re.split(' ', text) for text in results]

        return results

    #过滤掉texts中的不属于mod中的字符
    def text_filtering(self, texts, mod='[^ غاۆنڭۈبە#$ىكلۇقپئۋيھېژزرتسوخجفچدگمش]', clean_non_labelled=True):
        #texts:[[str1, str2...], [str1...], ...]
        #如果要去除#$，mod='[^ غاۆنڭۈبەىكلۇقپئۋيھېژزرتسوخجفچدگمش]'
        #clean_non_labelled为True时，文档中不存在#和$的词汇将会被去除，但保留#与$。
        #clean_non_labelled为False时，将对texts清洗，去除所有不属于mod中的字符。
        #return:[str0, str1, str2...]
        r = []
        if clean_non_labelled == False:
            mod='[^ غاۆنڭۈبەىكلۇقپئۋيھېژزرتسوخجفچدگمش]'
        for text in texts:
            for word in text:
                if (('#' in word) | ('$' in word) and clean_non_labelled):
                    r.append(re.sub(mod, ' ', word))
                elif clean_non_labelled == False:
                    temp = re.sub(mod, '', word)
                    if temp != '':
                        r.append(re.sub(mod, '', word))
                else:
                    continue
        return r


    #将texts中的每一个元素以mod中的元素进行分割
    def text_split(self, texts, mod='[#$ ]+'):
        #texts:[text0, text1...]
        r = []
        for text in texts:
            r.extend(re.split(mod, text))

        return r

    def get_seg(self):
        vocab = []
        for file_name in tqdm(self.files_list, desc='Processing'):
            texts = self.read_file(file_name)
            r = self.text_filtering(texts)
            vocab.extend(self.text_split(r))
        pass

        return vocab


    def get_words(self):
        words_list = []
        for file_name in tqdm(self.files_list, desc='Processing'):
            texts = self.read_file(file_name)
            r = self.text_filtering(texts, clean_non_labelled=False)
            words_list.extend(r)

        return words_list

    #将str形式的字符串word转为由char组成的列表
    def words_to_chars(self, words_list):
        char_list = []
        for word in tqdm(words_list, desc='Transforming'):
            char_list.append(' '.join(word).split(' '))
        
        return char_list

#为主模型提供数据
class Data():
    def __init__(self, path=data_path):
        self.path = path 
        self.train_files = [self.path+'train.src', self.path+'train.trg']
        self.test_files = [self.path+'test.src', self.path+'test.trg']
        self.dev_files = [self.path+'dev.src', self.path+'dev.trg']

        #自动构建词典
        #self.char2id_dict = {} 
        #self.word2id_dict = {}
        #手动定义
        self.class_id = dict(zip((tag_set+[PAD]), range(len(tag_set)+1)))
        self.class_to_id_fun = lambda x : self.class_id.get(x)
        self.vocab_dict, self.word_to_id_fun = self.dictionary_generator()
        self.vocab_ivdict = {v: k for k, v in self.vocab_dict.items()}
        pass 


    def load_batch_data(self, data_class = 'train', batch_size=None):
        #data_class: train/test/dev
        #返回填充数据：
        #当batch_size被指定时，返回一个由batch data组成的列表
        #batch_size未指定时，返回所有数据
        if data_class == 'train':
            data = self.load_data(*self.train_files)
        elif data_class == 'test':
            data = self.load_data(*self.test_files)
        elif data_class == 'dev':
            data = self.load_data(*self.dev_files)
        if batch_size is None:
            batch_size = len(data[0])
        print('loading ' + data_class + ' data...')
        X = []
        Y = []
        seq_len = []
        batch_num = (len(data[0]) - 1) // batch_size 
        for i in range(batch_num):
            #batch
            batch_x = data[0][i * batch_size : (i + 1) * batch_size]
            batch_y = data[1][i * batch_size : (i + 1) * batch_size]
            '''
            temp_len = [len(vector) for vector in batch_x]
            max_len = max(temp_len)
            seq_len.append(np.array(temp_len, dtype=np.int32))

            #填充数据和标签
            temp_x = np.full((batch_size, max_len), self.word_to_id_fun(PAD), np.int32)
            temp_y = np.full((batch_size, max_len), self.class_to_id_fun(PAD), np.int32)
            for j in range(batch_size):
                temp_x[j, :len(batch_x[j])] = batch_x[j]
                temp_y[j, :len(batch_y[j])] = batch_y[j]
            '''
            temp_x, _ = modules.pad(batch_x, self.word_to_id_fun(PAD))
            temp_y, temp_len = modules.pad(batch_y, self.class_to_id_fun(PAD))
            seq_len.append(np.array(temp_len, dtype=np.int32))
            X.append(np.array(temp_x))
            Y.append(np.array(temp_y))
        
        #按同样的逻辑处理原数据中不足batch_size的部分
        batch_x = data[0][batch_num * batch_size : ]
        batch_y = data[1][batch_num * batch_size : ]
        '''
        temp_len = [len(vector) for vector in batch_x]
        max_len = max(temp_len)
        seq_len.append(np.array(temp_len, dtype=np.int32))
        temp_x = np.full((len(batch_x), max_len), self.word_to_id_fun(PAD), np.int32)
        temp_y = np.full((len(batch_y), max_len), self.class_to_id_fun(PAD), np.int32)
            
        for j in range(len(batch_x)):
            temp_x[j, :len(batch_x[j])] = batch_x[j]
            temp_y[j, :len(batch_y[j])] = batch_y[j]
        '''
        temp_x, _ = modules.pad(batch_x, self.word_to_id_fun(PAD))
        temp_y, temp_len = modules.pad(batch_y, self.class_to_id_fun(PAD))
        seq_len.append(np.array(temp_len, dtype=np.int32))
        X.append(np.array(temp_x, dtype=np.int32))
        Y.append(np.array(temp_y, dtype=np.int32))
        
        
        print('done, batch_size: {}, batch_num of each epoch: {}, count of samples: {}'.format(str(batch_size), str(batch_num+1), str(len(data[0]))))       
        if batch_num != 0:
            return X, Y, seq_len
        else:
            print(len(X))
            return X[0], Y[0], seq_len

    #从file_x, file_y中读入数据，返回经过符号过滤的chars和labels
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
                x.append(x_lines[index])
                y.append(y_lines[index])
        return [x, y]

    #从目录中读取所有数据，构建char到Id的映射表，并返回映射表和映射函数
    def dictionary_generator(self):
        all_src = [] 
        for file_name in os.listdir(self.path):
            if file_name.split('.')[-1] == 'src':
                all_src.append(file_name)
        all_words = []
        for file_name in all_src:
            with open(self.path+file_name, 'r',encoding='utf-8-sig') as f:
                all_texts = [re.sub(r' |\ufeff', '', line.strip('\n')) for line in f.readlines()]
                for each_text in all_texts:
                    all_words.extend(each_text)

        word_count = Counter(all_words).most_common(100)
        words_set, _ = zip(*word_count)
        words_set += (PAD, )
        vocab_dict = dict(zip(words_set, range(len(words_set))))
        return vocab_dict, lambda x : vocab_dict.get(x, vocab_dict[PAD])
        
    def seg_decoder(self, data):
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
            #如果flag为1,表明正在组合某个seg
            if label == self.class_to_id_fun('b'):
                if flag == 1:
                    segment.append(seg)
                seg = str(x[index])
                flag=1
            if label == self.class_to_id_fun('m'):
                seg += str(x[index])
            if label == self.class_to_id_fun('e'):
                seg += str(x[index])
                flag = 0
                segment.append(seg)
                seg = ''
            if label == self.class_to_id_fun('s'):
                segment.append(str(x[index]))
                flag = 0
            
        return segment   
    

    def get_iv_oov(self, data_class):
        base_data = self.load_data(*self.train_files)
        if data_class == 'test':
            data = self.load_data(*self.test_files)
        elif data_class == 'dev':
            data = self.load_data(*self.dev_files)
        base_segment = set([])
        segment = set([])
        for x, y in zip(*base_data):
            base_segment |= set(self.seg_decoder([[self.vocab_ivdict[c] for c in x], y]))
        for x, y in zip(*data):
            segment |= set(self.seg_decoder([[self.vocab_ivdict[c] for c in x], y]))
        
        pass 
        return segment&base_segment, segment - base_segment


#建立char的word2vec
def build_word2vec(window, size, path='/home/hanshuo/Documents/Research/LATTICE_SEGMENTATION/data/word2vec_models'):
    DL = DataLoader()
    print('get all words from files...')
    words_list = DL.get_words()
    print('prepare for training...')
    char_list = DL.words_to_chars(words_list)
    print('training...')
    model = word2vec.Word2Vec(char_list, window=window, size=size, workers=8)
    
    model.save(path+'/'+str(window)+'_'+str(size)+'.word2vec')
    return model


def build_glove(window, size, path='/home/hanshuo/Documents/Research/LATTICE_SEGMENTATION/data/glove_models'):
    DL = DataLoader()
    print('get all words from files...')
    words_list = DL.get_words()
    print('prepare for training...')
    char_list = DL.words_to_chars(words_list)
    print('training...')
    corpus_model = Corpus()
    corpus_model.fit(char_list, window=window)

    glove = Glove(no_components=size, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=10, no_threads=8, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    glove.save(path+'/'+str(window)+'_'+str(size)+'.glove')
    #glove = Glove.load(path+'/'+str(window)+'_'+str(size)+'.glove')
    return glove
'''
'''




