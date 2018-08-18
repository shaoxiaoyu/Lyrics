import os
from collections import Counter
from itertools import chain
import configparser as cp

import numpy as np
import jieba
jieba.add_word("_space_")


def config_reader():
    """
    Parse model's parameter from configuration_file
    """
    config_dict = {}
    conf = cp.ConfigParser()
    conf.read('./datas/config.ini')

    config_dict['data_folder'] = str(conf.get('DATA', 'DATA_FOLDER'))
    config_dict['batch_size'] = int(conf.get('DATA', 'BATCH_SIZE'))
    config_dict['vocab_size'] = int(conf.get('DATA', 'VOCAB_SIZE'))
    config_dict['window_size'] = int(conf.get('DATA', 'WINDOW_SIZE'))
    config_dict['data_path'] = str(conf.get('DATA', 'DATA_PATH'))

    config_dict['embedding_dim'] = int(conf.get('MODEL', 'EMBEDDING_DIM'))
    config_dict['num_layers'] = int(conf.get('MODEL', 'NUM_LAYERS'))
    config_dict['num_utils'] = int(conf.get('MODEL', 'NUM_UTILS'))
    config_dict['FCNN_num_units'] = int(conf.get('MODEL', 'FCNN_NUM_UTILS'))
    config_dict['keep_prob'] = float(conf.get('MODEL', 'KEEP_PROB'))
    config_dict['rnn_mode'] = str(conf.get('MODEL', 'RNN_MODE'))
    config_dict['max_epoch'] = int(conf.get('MODEL', 'MAX_EPOCH'))
    config_dict['learning_rate'] = float(conf.get('MODEL', 'LEARNING_RATE'))
    config_dict['end_loss'] = float(conf.get('MODEL', 'END_LOSS'))

    config_dict['model_path'] = str(conf.get('MODEL', 'MODEL_PATH'))
    config_dict['rhyme_path'] = str(conf.get('MODEL', 'RHYME_PATH'))
    config_dict['logs_file'] = str(conf.get('MODEL', 'LOGS_FILE'))
    config_dict['save_model_name'] = str(conf.get('MODEL', 'SAVE_MODEL_NAME'))
    config_dict['print_step'] = int(conf.get('MODEL', 'PRINT_STEP'))
    config_dict['save_epoch'] = int(conf.get('MODEL', 'SAVE_EPOCH'))

    conf.clear()

    return config_dict


class DataUtils(object):

    def __init__(self, config=config_reader()):
        self.data_folder = config['data_folder']
        self.batch_size = config['batch_size']
        self.vocab_size = config['vocab_size']
        self.window_size = config['window_size']
        self.data_path = config['data_path']
        self.sentence_list, self.topic_list, self.cur_sentences, self.cur_topic, self.next_sentences = self._load_data()
        self.vocab, self.word_to_int, self.int_to_word, self.topic_to_int, self.int_to_topic = self._get_vocab()
        self.chunk_size = len(self.cur_sentences) // self.batch_size

    def _load_data(self):
        """
        Load data and construct the training samples
        :return:
        """
        sentence_list = []
        topic_list = []
        with open(self.data_path, "r") as f:
            line = f.readline().strip()
            while line:
                line_list = line.split(",")
                sen = line_list[0].split()
                sentence_list.append(sen)
                topic_list.append(line_list[1])
                line = f.readline().strip()

        cur_sentences = []
        cur_topic = []
        next_sentences = []
        for i in range(len(sentence_list) - 1):
            if topic_list[i] == topic_list[i+1]:
                cur_sentences.append(sentence_list[i])
                next_sentences.append(sentence_list[i+1])
                cur_topic.append(topic_list[i])
        return sentence_list, topic_list, cur_sentences, cur_topic, next_sentences

    def _get_vocab(self):
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        total_words = list(chain.from_iterable(self.sentence_list))

        # The top most_common words
        word_count = Counter(total_words)
        word_count = word_count.most_common(self.vocab_size)
        word_count = {w: c for w, c in word_count}
        split_words = word_count.keys()
        vocab = sorted(set(split_words)) + special_words
        word_to_int = {w: index for index, w in enumerate(vocab)}
        int_to_word = {index: w for index, w in enumerate(vocab)}

        # Build topics table
        topic_set = set(self.topic_list)
        topic_to_int = {w: index for index, w in enumerate(topic_set)}
        int_to_topic = {index: w for index, w in enumerate(topic_set)}
        return vocab, word_to_int, int_to_word, topic_to_int, int_to_topic

    def get_sen_index(self, sentence):
        sentence = sentence[::-1]
        sentence_to_index = [self.word_to_int.get(word, self.word_to_int['<UNK>'])for word in sentence]
        return sentence_to_index

    def get_batch_length(self, batch_index):
        batch_len = [len(sen) for sen in batch_index]
        return batch_len

    def batch_padding(self, batch_index):
        max_length = max(self.get_batch_length(batch_index))
        batch_size = len(batch_index)
        batch_padding = np.full((batch_size, max_length), self.word_to_int['<PAD>'], np.int32)
        for row in range(batch_size):
            batch_padding[row, :len(batch_index[row])] = batch_index[row]
        return batch_padding

    def prepare_batch(self):

        start, end = 0, self.batch_size
        for _ in range(self.chunk_size):
            # prepare cur_sentences encode
            batch_cur_sentences = self.cur_sentences[start:end]
            encode = [self.get_sen_index(sen) for sen in batch_cur_sentences]
            encode_length = self.get_batch_length(encode)
            encode = self.batch_padding(encode)

            # prepare topic decode
            decode_topic_label = [self.topic_to_int[topic] for topic in self.topic_list[start:end]]

            # prepare next_sentences decode
            batch_next_sentences = self.next_sentences[start:end]
            decode_sen_input = [self.get_sen_index(['<GO>']) + self.get_sen_index(sen) for sen in batch_next_sentences]
            decode_sen_label = [self.get_sen_index(sen) + self.get_sen_index(['<EOS>']) for sen in batch_next_sentences]
            decode_sen_length = self.get_batch_length(decode_sen_input)
            decode_sen_input = self.batch_padding(decode_sen_input)
            decode_sen_label = self.batch_padding(decode_sen_label)

            yield encode, decode_topic_label, decode_sen_input, decode_sen_label, encode_length, decode_sen_length

            start += self.batch_size
            end += self.batch_size


if __name__ == '__main__':
    data_utils = DataUtils()
    batch = data_utils.prepare_batch()
    for i in batch:
        print(1)
