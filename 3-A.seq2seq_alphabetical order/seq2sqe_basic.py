#coding=utf-8
#Author:Dodo
#Date:2019-02-03
#Blog:www.pkudodo.com

'''
文件结构:
  sseq2sqe_basic.py:训练模型
  test.py:结果测试

训练结果：
#input: [8, 5, 12, 12, 15]
output: [ 5  8 12 12 15]
-----------------------
the input is: hello
the output is: ['e', 'h', 'l', 'l', 'o']
'''
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

def loadData(fileName, fileType='source'):
    '''
    加载文件
    :param fileName:文件名
    :param fileType: source：源文件   target：目标文件
    :return:
    '''
    # 载入文本，并转换为小写
    text = open(fileName).read().lower()

    #遍历文本所有内容，将单词抽取出来生成词汇表
    vocab_list = list(set(word for word in text))
    #对词汇表进行排序
    #因为在生成词汇表的时候使用了集合，不能保证每次运行得到的词汇顺序一直
    #所以要进行排序，固定词汇顺序，保证后续不同运行时生成的单词和其对应索引一致
    vocab_list.sort()

    #特殊字符，分别为GO：一个文本的开始   EOS：一个文本的技术   PAD：文本填充字符  UNK：未知字符
    special_vocab = ['<GO>', '<EOS>', '<PAD>', '<UNK>']
    #生成单词 -》 索引的字典
    vocab2int = {word:index for index, word in enumerate(vocab_list + special_vocab)}
    #生成索引 -》 单词的字典
    int2vocab = {index:word for word, index in vocab2int.items()}

    if fileType=='source':
        #如果是源文件，就直接将单词转换为索引即可
        text = [[vocab2int[char] for char in line] for line in text.split('\n')]
    else:
        #如果是目标文件，在训练过程中需要添加EOS表示文本结束
        text = [[vocab2int[char] for char in line] + [vocab2int['<EOS>']] for line in text.split('\n')]

    return text, vocab2int, int2vocab

def get_input():
    '''
    创建输入tensor
    :return:
    '''
    #源文件输入
    input_data = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder-input')
    #目标文件输出
    target = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder-output')  ####################################

    #编码层序列长度    也就是源文本每个文本的长度
    encoder_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='encoder_sequence_length')
    #解码层序列长度    也就是原文本对应的目标文本的长度
    decoder_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='decoder_sequence_length')
    #解码层最大长度    就是一个batch内目标文本标签中的最大长度
    decoder_sequence_max_length = tf.reduce_max(decoder_sequence_length)

    return input_data, target, encoder_sequence_length, decoder_sequence_length, decoder_sequence_max_length

def get_rnn_cell(rnn_size):
    '''
    创建单层lstm
    :param rnn_size:
    :return:
    '''
    cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    return cell


train_graph = tf.Graph()
def create_model(vocab_size, embeding_size, rnn_size, num_layers, vocab2int, batch_size, learn_rate):
    '''
    创建模型
    :param vocab_size:词汇表大小
    :param embeding_size: 词嵌入矩阵中表示一个单词所用的维度大小
    :param rnn_size: lstm节点中隐层节点数量
    :param num_layers: lstm层数
    :param vocab2int: 字典   单词 -》 索引
    :param batch_size: 一个batch的大小
    :param learn_rate: 学习率
    :return:
    '''
    with train_graph.as_default():
        #获取输入
        input_data, target, encoder_sequence_length, decoder_sequence_length, decoder_sequence_max_length\
            = get_input()

        #随机生成词嵌入矩阵，矩阵内元素在训练过程中会得到训练
        #实际上也可网上下载现成的词嵌入矩阵直接用于训练会更好，在本项目中主要的学习重点不在这里，所以就不整了
        #下一个自动生成摘要的项目中，使用的是现成的词嵌入矩阵
        embedings = tf.Variable(tf.random_normal(shape=(vocab_size, embeding_size), stddev=0.1))
        #使用embedding_lookup在词嵌入矩阵中找到单词所对应的单词向量
        encode_input = tf.nn.embedding_lookup(embedings, input_data, name='encode_input')

        #创建多层lstm
        cell = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell(rnn_size) for _ in range(num_layers)])
        #提交
        #encoder_final_state是最后一个节点输出的状态，也就是解码中的输入
        _, encoder_final_state = tf.nn.dynamic_rnn(cell, encode_input, sequence_length=encoder_sequence_length, dtype=tf.float32)

        #decode分为训练和预测两种模式
        #可以把lstm想象成一条链，每一个节点的输出是下一个节点的输入，但是在训练过程中可能会出现一个问题：例如当第一个节点
        #刚开始训练的时候，可能输出的是错的，那么第二个节点的输入就已经错了，很难再输出对的，没有办法进行一个有效的训练
        #所以在seq2seq模型中，模型的decode通常会分为两部分
        #在训练过程中，配合标签进行使用，每一个lstm节点的输出不再作为下一个节点的输入，而是将标签作为下一个节点的输入，
        #这样保证在训练过程中每一个节点的输入都是正确的，每一个节点都能得到有效的训练。
        #在预测过程中，节点的输出再次作为下一个节点的输入

        #在训练过程中，需要给定标签，作为下一个节点的输入
        #但是decode层中第一个节点的输入，应该是GO，表示一个文本开始了
        #所以我们需要现在手动在batch内每个文本的开头添加一个GO标识
        #此外最后一个节点的输出，不再作为下一个节点的输入了，因为后面没有节点了，所以直接把最后一个编码去掉就可以了
        #（实际上最后一个节点要么是EOS，要么是PAD，本身也是无意义信息）

        #裁剪掉每个标签的最后一个编码
        ending = tf.strided_slice(target, [0, 0], [batch_size, -1], [1, 1])
        #在每个标签的开头添加GO
        decode_input = tf.concat([tf.fill([batch_size, 1], vocab2int['<GO>']), ending], 1)

        #根据给出的文本编码，在词嵌入矩阵中找到对应的词向量
        decoder_train_input = tf.nn.embedding_lookup(embedings, decode_input, name='decoder_train_input')

        #构建decode，由num_layers层lstm组成
        cell = tf.nn.rnn_cell.MultiRNNCell([get_rnn_cell(rnn_size) for _ in range(num_layers)])
        #在decode中每个lstm的输出都是一个向量，在这个向量上添加全连接层，实际上后面在这个全连接层后面还有
        #softmax层，最终输出概率最大的单词索引
        output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        #训练的decode
        #训练和下面的预测的decode是在同一个命名空间中，是希望训练和预测共用内部的所有权值
        with tf.variable_scope('decode'):
            #构建helper，告诉系统将标签作为每一个节点的输入
            decode_train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_train_input, decoder_sequence_length)
            #生成decode
            train_basic_decoder = tf.contrib.seq2seq.BasicDecoder(cell, decode_train_helper, encoder_final_state, output_layer)
            #提交
            train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_basic_decoder, impute_finished=True, maximum_iterations=decoder_sequence_max_length)

        #预测用的decode
        with tf.variable_scope('decode', reuse=True):
            #在下方的helper中需要告诉系统起始符号是什么，要求是一个tensor，所以这里定义一个start_tokens
            start_tokens = tf.tile(tf.constant([vocab2int['<GO>']], dtype=tf.int32), [batch_size])
            #生成helper，告诉系统每一个节点的输出作为下一个节点的输入
            decode_test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedings, start_tokens, vocab2int['<EOS>'])
            #生成decode
            test_basic_decoder = tf.contrib.seq2seq.BasicDecoder(cell, decode_test_helper, encoder_final_state, output_layer)
            #提交
            test_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(test_basic_decoder, impute_finished=True, maximum_iterations=decoder_sequence_max_length)

        #：final_outputs是tf.contrib.seq2seq.BasicDecoderOutput类型，包括两个字段：rnn_output，sample_id。
        #rnn_output是输出的向量，是词汇表大小的
        #sample_id是上面的输出中找到值最大的（也就是概率最大），返回的那个索引
        #训练的decode的输出结果，是一个词汇表大小的向量
        train_logits = tf.identity(train_decoder_output.rnn_output, name='logits')
        #预测输出的结果，是一个具体的索引值
        #（实际上也是一个向量，输出的字符应该多长，这里的编码就多长，每一个位置放的是该位置字符的索引）
        #例如输入cba， 输出应该是abc
        #train_logits输出的可能是是  [0.8, 0.1, 0,05, 0, 0. ....], [0.05, 0.6, 0.3, 0, 0. ...], [0.01, 0.2, 0.7, 0, ....]
        #predict_logits输出的可能是 [0, 1, 2]   分别表示 [a， b， c]
        predict_logits = tf.identity(test_decoder_output.sample_id, name='predictions')

        #tf.sequence_mask的用法百度能更说的清一点，建议百度一下
        #实际上作的就是对于每一个输出来说，要计算损失，但是一般文本都或多或少会有一些pad填充项，mask的作用是为了让
        #pad的位置不参与loss的计算，比如说decode的输出长度时10，文本实际长度时5，那么计算损失的时候，就只计算前5项
        mask = tf.sequence_mask(decoder_sequence_length, decoder_sequence_max_length, dtype=tf.float32)
        with tf.variable_scope('optimizer'):
            #计算损失，
            #在计算损失过程中，softmax需要进行一个vacab_size大小的求和计算，不方便
            #tf.contrib.seq2seq.sequence_loss在softmax的时候，会获取原数据量中的一个子集来进行softmax，
            #这个子集一般正样本，一般负样本，减少计算量
            loss = tf.contrib.seq2seq.sequence_loss(train_logits, target, mask)
            #设置学习率
            optimizer = tf.train.AdamOptimizer(learn_rate)

            #梯度裁剪，防止梯度爆炸
            #对loss求导，获得训练变量的梯度
            gardiences = optimizer.compute_gradients(loss)
            #将梯度裁剪大-5到5
            new_gardiences = [(tf.clip_by_value(gards, -5, 5), val) for gards, val in gardiences if gards is not None]
            #更新梯度
            train_op = optimizer.apply_gradients(new_gardiences)
        return input_data, target, encoder_sequence_length, decoder_sequence_length, train_op, loss


def get_batch(input_data, target, batch_size, vocab2int):
    '''
    获取一个batch
    :param input_data:输入文本
    :param target: 输出文本
    :param batch_size: 一个batch的大小
    :param vocab2int: 字典   单词 -》 索引
    :return:
    '''

    #先判断有多少个batch
    n_batch = len(input_data) // batch_size

    #遍历
    for i in range(0, (n_batch * batch_size - 1), batch_size):
        #获取单个batch数据
       batch_input = input_data[i: i + batch_size]
       batch_target = target[i: i + batch_size]

        #得到每一个输入文本的长度
       input_length = []
       for sample in batch_input:
           input_length.append(len(sample))

        #得到每一个输出文本的长度
       target_length = []
       for sample in batch_target:
           target_length.append(len(sample))

        #计算输入文本中的最大值
       input_max_length = max(input_length)
        #依据最大值，将所有输入文本进行一个填充
       batch_input = np.array([sample + [vocab2int['<PAD>']] * (input_max_length - len(sample)) for sample in batch_input], dtype=np.int32)

        #计算输出文本的最大值
       target_max_length = max(target_length)
        #依据最大值，将输出文本进行一个填充
       batch_target = np.array([sample + [vocab2int['<PAD>']] * (target_max_length - len(sample)) for sample in batch_target], dtype=np.int32)

        #百度yield，这是构造一个生成器
       yield batch_input, input_length, batch_target, target_length

epochs = 50
batch_size = 128
embeding_size = 20
rnn_size = 128
num_layers = 2
learn_rate = 0.001

checkpoint = "checkpoint/model"

def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

if __name__ == '__main__':
    #加载源文本
    source_text,  vocab2int, int2vocab = loadData('data/letters_source.txt', 'source')
    #加载目标文本
    target_text,  _, _ = loadData('data/letters_target.txt', 'target')

    #创建模型
    input_data, target, encoder_sequence_length, decoder_sequence_length, train_op, loss = \
        create_model(len(vocab2int), embeding_size, rnn_size, num_layers, vocab2int, batch_size, learn_rate)


    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        count = 0
        for epoch in range(epochs):
            #每次获取一个batch
            for (batch_input, batch_input_length, batch_target, batch_target_length) \
                    in get_batch(source_text, target_text, batch_size, vocab2int):
                count += 1
                feed = {
                    input_data:batch_input,
                    target:batch_target,
                    encoder_sequence_length:batch_input_length,
                    decoder_sequence_length:batch_target_length
                }

                #喂数据，运行
                batch_loss, _ = sess.run([loss, train_op], feed_dict=feed)

                #定期打印
                if count % 50 == 0:
                    print('epoch:%d/%d, count:%d'%(epoch, epochs, count))
                    print('train loss:%.4f'%(batch_loss))

        saver.save(sess, checkpoint)



