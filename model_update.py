import tensorflow as tf
import tensorflow.contrib.seq2seq as sq
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from data_utils import DataUtils, config_reader
from tqdm import tqdm

class Model(object):
    def __init__(self, config=config_reader()):
        """
        read model param
        """
        self.rnn_mode = config['rnn_mode']
        self.batch_size = config['batch_size']
        self.embedding_dim = config['embedding_dim']
        self.num_layers = config['num_layers']
        self.num_units = config['num_utils']
        self.FCNN_num_units = config['FCNN_num_units']
        self.learning_rate = config['learning_rate']
        self.max_epoch = config['max_epoch']
        self.keep_prob = config['keep_prob']
        self.model_path = config['model_path']
        self.logs_file = config['logs_file']
        self.end_loss = config['end_loss']
        self.save_model_name = config['save_model_name']
        self.print_step = config['print_step']
        self.save_epoch = config['save_epoch']

        self.data_utils = DataUtils()
        self.vocab = self.data_utils.vocab
        self.chunk_size = self.data_utils.chunk_size
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)

    @staticmethod
    def soft_max_variable(num_units, vocab_size, reuse=False):
        with tf.variable_scope('soft_max', reuse=reuse):
            w = tf.get_variable("w", [num_units, vocab_size])
            b = tf.get_variable("b", [vocab_size])
        return w, b

    @staticmethod
    def build_inputs():
        with tf.variable_scope('inputs'):
            # cur_sentences encode
            encode = tf.placeholder(tf.int32, shape=[None, None], name='encode')
            encode_length = tf.placeholder(tf.int32, shape=[None, ], name='encode_length')

            # topic decode
            decode_topic_label = tf.placeholder(tf.int32, shape=[None, ], name='decode_topic_label')

            # next_sentences decode
            decode_sen_input = tf.placeholder(tf.int32, shape=[None, None], name='decode_sen_input')
            decode_sen_label = tf.placeholder(tf.int32, shape=[None, None], name='decode_sen_label')
            decode_sen_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_sen_length')

        return encode, decode_topic_label, decode_sen_input, decode_sen_label, encode_length, decode_sen_length, decode_sen_input

    def build_word_embedding(self, encode, decode_sen_input):
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable(name='embedding',
                                        shape=[len(self.vocab), self.embedding_dim],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))
            encode_emb = tf.nn.embedding_lookup(embedding, encode, name='encode_emb')
            decode_sen_input_emb = tf.nn.embedding_lookup(embedding, decode_sen_input, name='decode_sen_input_emb')
        return encode_emb, decode_sen_input_emb

    def build_encoder(self, encode_emb, encode_length, scope='encoder', train=True):
        batch_size = self.batch_size if train else 1

        if self.rnn_mode == 'Bi-directional':
            with tf.variable_scope(self.rnn_mode + scope):
                LSTM_fw_cell = rnn.BasicLSTMCell(num_units=self.num_units)
                LSTM_bw_cell = rnn.BasicLSTMCell(num_units=self.num_units)
                # dropout
                LSTM_fw_cell = rnn.DropoutWrapper(LSTM_fw_cell, output_keep_prob=self.keep_prob)
                LSTM_bw_cell = rnn.DropoutWrapper(LSTM_bw_cell, output_keep_prob=self.keep_prob)
                # initial
                initial_state_fw = LSTM_fw_cell.zero_state(batch_size, tf.float32)
                initial_state_bw = LSTM_bw_cell.zero_state(batch_size, tf.float32)
                initial_state_c = tf.reduce_mean(tf.stack([initial_state_fw.c, initial_state_bw.c], axis=0), axis=0)
                initial_state_h = tf.reduce_mean(tf.stack([initial_state_fw.h, initial_state_bw.h], axis=0), axis=0)
                initial_state = rnn.LSTMStateTuple(initial_state_c, initial_state_h)

                # inputs:[batch_size,n_steps,n_input] outputs, output_states
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_fw_cell,
                                                                         cell_bw=LSTM_bw_cell,
                                                                         initial_state_fw=initial_state_fw,
                                                                         initial_state_bw=initial_state_bw,
                                                                         inputs=encode_emb,
                                                                         sequence_length=encode_length,
                                                                         dtype=tf.float32
                                                                         )

                # output_states is a tuple like (fw's {LSTMStateTuple},bw's{LSTMStateTuple})
                # LSTMStateTuple is (c=(?, num_units),h=(?, num_units))
                state_c = tf.reduce_mean(tf.stack([output_states[0].c, output_states[1].c], axis=0), axis=0)
                state_h = tf.reduce_mean(tf.stack([output_states[0].h, output_states[1].h], axis=0), axis=0)
                state_final = rnn.LSTMStateTuple(state_c, state_h)

                outputs = tf.concat(outputs, axis=-1)
            return initial_state, outputs, state_final

        elif self.rnn_mode == 'Multilayer-RNN ':
            with tf.variable_scope(self.rnn_mode+scope):
                # build two layers LSTM
                stack_rnn = []
                for i in range(self.num_layers):
                    cell = rnn.BasicLSTMCell(num_units=self.num_units)
                    drop_cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
                    stack_rnn.append(drop_cell)
                cell = rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
                initial_state = cell.zero_state(batch_size, tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                         inputs=encode_emb,
                                                         initial_state=initial_state,
                                                         sequence_length=encode_length)

                outputs = tf.concat(outputs, axis=-1)
            return initial_state, outputs, final_state

        else:
            pass

    def nn_layer(self, input_tensor, input_dim, output_dim, act=tf.nn.relu):
        w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.01, shape=[output_dim]), name="b")
        logits = tf.matmul(input_tensor, w) + b
        activations = act(logits, name='activation')
        return activations

    def build_decoder_topic(self, state_input, keep_prob):

        hidden = self.nn_layer(input_tensor=state_input,
                               input_dim=self.num_units,
                               output_dim=self.FCNN_num_units)

        dropout = tf.nn.dropout(hidden, keep_prob, name="dropout")

        logits = self.nn_layer(input_tensor=dropout,
                               input_dim=self.FCNN_num_units,
                               output_dim=len(self.data_utils.int_to_topic),
                               act=tf.identity)
        return logits

    def build_decoder_topic_loss(self, logits, true_labels):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=true_labels, logits=logits)
        return cross_entropy

    def build_decoder_topic_accuracy(self, logits, true_labels):
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), true_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def build_decoder_sen(self, decode_emb, decode_inputs, length, state, memory, scope='next_sen_decoder', reuse=False):

        if self.rnn_mode == 'Bi-directional':
            with tf.variable_scope(self.rnn_mode + scope):
                cell = rnn.BasicLSTMCell(num_units=self.num_units)
                attn_mech = tf.contrib.seq2seq.LuongAttention(
                    num_units=self.embedding_dim,
                    memory=memory,
                    memory_sequence_length=length - 1,
                    name='LuongAttention')
                dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=cell,
                    attention_mechanism=attn_mech,
                    attention_layer_size=self.embedding_dim,
                    name='Attention_Wrapper')
                initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size).clone(
                                                    cell_state=state)
                output_layer = Dense(len(self.vocab), name='output_projection')
                max_dec_len = tf.reduce_max(length, name='max_dec_len')
                training_helper = sq.TrainingHelper(
                    inputs=decode_emb,
                    sequence_length=length,
                    time_major=False,
                    name='training_helper')
                training_decoder = sq.BasicDecoder(
                    cell=dec_cell,
                    helper=training_helper,
                    initial_state=initial_state,
                    output_layer=output_layer)
                train_dec_outputs, train_dec_last_state, _ = sq.dynamic_decode(
                    training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_dec_len)
                logits = tf.identity(train_dec_outputs.rnn_output, name='logits')
                targets = tf.slice(decode_inputs, [0, 0], [-1, max_dec_len], 'targets')
                masks = tf.sequence_mask(length, max_dec_len, dtype=tf.float32, name='masks')
                batch_loss = sq.sequence_loss(
                    logits=logits,
                    targets=targets,
                    weights=masks,
                    name='batch_loss')
                valid_predictions = tf.identity(train_dec_outputs.sample_id, name='valid_preds')
                return batch_loss, valid_predictions, train_dec_last_state

        elif self.rnn_mode == 'Multilayer_':
            with tf.variable_scope(self.rnn_mode+scope):
                stack_rnn = []
                for i in range(self.num_layers):
                    cell = rnn.BasicLSTMCell(num_units=self.num_units)
                    drop_cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                    stack_rnn.append(drop_cell)
                cell = rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
                outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                         inputs=decode_emb,
                                                         initial_state=state,
                                                         sequence_length=length)
            x = tf.reshape(outputs, [-1, self.num_units])
            w, b = self.soft_max_variable(self.num_units, len(self.vocab), reuse=reuse)
            logits = tf.matmul(x, w) + b
            prediction = tf.nn.softmax(logits, name='predictions')
            return logits, prediction, final_state

        else:
            pass

    def build_decoder_sen_loss(self, logits, targets, scope='loss'):
        with tf.variable_scope(scope):
            y_one_hot = tf.one_hot(targets, len(self.vocab))
            y_reshaped = tf.reshape(y_one_hot, [-1, len(self.vocab)])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        return loss

    def build_optimizer(self, loss, scope='optimizer'):
        with tf.variable_scope(scope):
            grad_clip = 5
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    def build(self, train=True):
        # input placeholder
        encode, decode_topic_label, decode_sen_input, decode_sen_label, encode_length, decode_sen_length, decode_inputs = self.build_inputs()

        # embedding
        encode_emb, decode_sen_input_emb = self.build_word_embedding(encode, decode_sen_input)

        # build encoder
        initial_state, memeory, final_state = self.build_encoder(encode_emb, encode_length, train=train)

        # build topic decoder
        state_input = final_state.h
        decoder_topic_logits = self.build_decoder_topic(state_input, self.keep_prob)
        decode_topic_loss = self.build_decoder_topic_loss(decoder_topic_logits, decode_topic_label)
        decode_topic_optimizer = self.build_optimizer(decode_topic_loss, scope='decode_topic_op')
        decode_topic_accuracy = self.build_decoder_topic_accuracy(decoder_topic_logits, decode_topic_label)

        # build post sentence decoder
        decode_sen_loss, decode_sen_prediction, decode_sen_state = self.build_decoder_sen(decode_sen_input_emb,
                                                                                          decode_inputs,
                                                                                          decode_sen_length,
                                                                                          final_state,
                                                                                          memeory,
                                                                                          scope='decoder_sen',
                                                                                          reuse=False)

        # decode_sen_loss = self.build_decoder_sen_loss(decode_sen_logits, decode_sen_label, scope='decoder_sen_loss')
        decode_sen_optimizer = self.build_optimizer(decode_sen_loss, scope='decoder_sen_op')

        inputs = {'encode': encode,
                  'decode_topic_label': decode_topic_label,
                  'decode_sen_input': decode_sen_input,
                  'decode_sen_label': decode_sen_label,
                  'encode_length': encode_length,
                  'decode_sen_length': decode_sen_length,
                  'initial_state': initial_state,
                  }

        decode_topic = {'decode_topic_optimizer': decode_topic_optimizer,
                        'decode_topic_loss': decode_topic_loss,
                        'decode_topic_accuracy': decode_topic_accuracy,
                        'decoder_topic_logits': decoder_topic_logits
                        }

        decode_sen = {'decode_sen_optimizer': decode_sen_optimizer,
                      'decode_sen_loss': decode_sen_loss,
                      'decode_sen_state': decode_sen_state,
                      'decode_sen_prediction': decode_sen_prediction,
                      }

        return inputs, decode_topic, decode_sen

    @staticmethod
    def restore(sess, saver, path):
        saver.restore(sess, save_path=path)
        print('Model restored from {}'.format(path))