#coding=utf-8
#Author:Dodo
#Date:2019-02-03
#Blog:www.pkudodo.com

'''
文件结构:
  harry_potter_lstm.py:训练模型
  generate_text.py:生成文本

训练结果：（给定首字母"Hi, "）
Hi, he was nearly off at Harry to say the time that and she had been back to his staircase of the too the Hermione?
'''

import tensorflow as tf
import numpy as np

def loadData(fileName):
    '''
    加载数据
    :param fileName:加载的文件名
    :return:
        vocab：文本包含的所有字符集合
        vocab2Int：字符到编码的映射
        int2Vocab：编码到字符的映射
        encode：编码后的文本（用索引表示整个文本）
    '''
    #读取文件
    test = open(fileName, encoding='utf-8').read()

    #将所有出现过的字符放入集合中，便于生成索引
    vocab = set(test)
    #通过排序保证每次运行后生成的索引一致
    vocabList = list(vocab)
    vocabList.sort()

    #词->索引
    vocab2Int = {word:index for index, word in enumerate(vocabList)}
    #索引->词
    int2Vocab = {index:word for word, index in vocab2Int.items()}
    #将文件编码
    encode = np.array([vocab2Int[word] for word in test])

    return vocab, vocab2Int, int2Vocab, encode

def get_batch(input_data, n_seqs, n_sequencd_length):
    '''
    将文件拆分成多个batch
    :param input: 待拆分文件
    :param n_seqs: 每个batch中包含的样本数目（也可以称之为句子数目）
    :param n_sequencd_length: 每个样本长度（包含的字符个数）
    :return:
    '''
    #一个batch的size是句子数目*句子长度
    batch_size = n_seqs * n_sequencd_length
    #计算需要多少个batch
    batch_num = len(input_data) // batch_size

    #将最后不能凑整的一些尾巴扔掉
    input_data = input_data[: batch_num * batch_size]
    #重新定义形状，将input变成一共n_seqs行的二维数组
    #获取单个batch流程：
    #   想象成一个方方正正的大饼，一共有n_seqs行，每次一个batch的尺寸是n_seqs * n_sequencd_length，
    #   所以每次取的时候，只需要在横坐标上量出n_sequencd_length的长度，然后竖着切一刀，尺寸就是一个batch
    #   size的大小，所以reshape的时候，行设置成n_seqs
    input_data = input_data.reshape((n_seqs, -1))

    for i in range(0, input_data.shape[1], n_sequencd_length):
        #按照上面的说法，矩阵中每次竖着切下一条batch
        x = input_data[:, i: i + n_sequencd_length]
        y = np.zeros_like(x)

        #lstm中前一个输出y是后一个输入x，所以x和y实际上是错开一位的
        #下面就是将x中的前移一位，转换成y
        y[:, : -1] = x[:, 1:]
        y[:, -1] = x[:, 0]

        #yield是一个生成器，百度有详细说明
        yield x, y

def model_input(n_seqs, n_sequencd_length):
    '''
    模型输入部分
    :param n_seqs:每次输入的样本数目，就是batch_size
    :param n_sequencd_length: 每个样本的长度
    :return:
    '''
    #初始化两个的占位
    #输入的大小为样本数目*样本长度
    #因为每个字符会对应一个字符的输出，所以target与input大小一致
    input = tf.placeholder(dtype=tf.int32, shape=(n_seqs, n_sequencd_length), name='input')
    target = tf.placeholder(dtype=tf.int32, shape=(n_seqs, n_sequencd_length), name='target')

    return input, target

def model_lstm(lstm_num_units, keep_prob, num_layers, n_seqs):
    '''
    构建lstm
    :param lstm_num_units:每个lstm节点内部的隐层节点数目
    :param keep_prob: drop比例
    :param num_layers: 层数目
    :param n_seqs: 每次传入多少样本
    :return:
    '''

    #在lstm的构建上，老版本的TensorFlow是这样创建lstm层的：
    #   初始化一个lstm节点，给节点添加drop，然后这个节点转换成列表传入MultiRNNCell就可以了
    #但是在我的版本上，这样子会报错，所以修改为：
    #   创建一个列表，如果我要创建3层lstm，那么我就要初始化3个lstm节点，然后放入列表中，再MultiRNNCell生成
    #猜测可能是底层代码发生了变化，老版本的方式是生成一个节点，然后节点不断复制。我的版本中可能节点复制后仍然是同一
    #节点，导致报错，所以每次都生成新的节点，然后再放入list中，保证所有节点都是唯一的。
    #以上是我猜测的，也可以联系我，大家一起讨论

    #创建列表，后续生成的节点都会放在列表里
    lstms = []

    #循环创建层
    for i in range(num_layers):
        #单独创建一层lstm节点
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units)
        #添加drop
        drop = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        #将节点放入list中
        lstms.append(drop)

    #创建lstm
    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
    #初始化输入状态
    init_state = cell.zero_state(n_seqs, dtype=tf.float32)      ##############################

    return cell, init_state

def model_output(lstm_output, in_size, out_size):
    '''
    模型输出
    lstm输出后在这里再通过softmax运算

    因为在输入时输入矩阵是n_seqs * n_sequencd_length大小的，lstm的隐层节点数目是lstm_num_units，
    所以lstm输出的大小是[n_seqs, n_sequencd_length, lstm_num_units]
    Softmax的大小是词汇表的长度，也就是len(vocab)
    在输出中一共有n_seqs * n_sequencd_length个字符，然后我们需要转换成[n_seqs * n_sequencd_length, lstm_num_units]，
    再通过softmax层，softmax是lstm_num_units，所以中间的w大小应该是[lstm_num_units, len(vocab)]

    首先需要做的是讲lstm的输出的维度转换成[n_seqs * n_sequencd_length, lstm_num_units]，变成二维的，再进行后续处理
    :param lstm_output: lstm的输出， 在这里是len(vocab)
    :param in_size: lstm的输出大小，在这里是lstm_num_units
    :param out_size:
    :return:
    '''

    #将维度转换为[n_seqs * n_sequencd_length, lstm_num_units]
    lstm_output = tf.reshape(lstm_output, shape=(-1, in_size))

    with tf.variable_scope('softmax'):
        #创建w和b
        softmax_w = tf.Variable(tf.truncated_normal(shape=(in_size, out_size), stddev=0.1), dtype=tf.float32, name='softmax_w')
        softmax_b = tf.Variable(tf.zeros(shape=(out_size)), dtype=tf.float32, name='softmax_b')

    #计算输出
    logits = tf.matmul(lstm_output, softmax_w) + softmax_b
    #计算输出的softmax
    output = tf.nn.softmax(logits)

    #返回
    return output, logits

def model_loss(target, logits, num_class):
    '''
    计算交叉熵损失
    :param target:标签
    :param logits: 预测输出
    :param num_class: 字符的种类数目
    :return:
    '''
    #将标签生成为onehot向量
    y_one_hot = tf.one_hot(target, num_class)

    #计算损失
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
    loss = tf.reduce_mean(loss)

    return loss

def model_optimizer(learning_rate, loss, clip_val):
    '''
    梯度裁剪，防止梯度爆炸
    :param learning_rate:学习率
    :param loss: 损失
    :param clip_val: 裁剪梯度
    :return:
    '''
    #设置学习率
    tran_op = tf.train.AdamOptimizer(learning_rate=learning_rate)

    #获取所有的训练变量
    allTvars = tf.trainable_variables()
    #loss对所有的训练变量进行求导，并进行梯度裁剪
    all_grads, _ = tf.clip_by_global_norm(tf.gradients(loss, allTvars), clip_norm=clip_val)
    #将裁剪后的梯度重新应用于所有变量
    optimizer = tran_op.apply_gradients(zip(all_grads, allTvars))

    return optimizer

class char_RNN:
    def __init__(self, vocab, n_seqs = 10, n_sequencd_length = 30, lstm_num_units=128, keep_prob=0.5, num_layers=3,
                 learning_rate=0.01, clip_val=5):
        '''
        初始化模型
        :param vocab:字符集合
        :param n_seqs: 每次训练的样本数目，也就是batch_size
        :param n_sequencd_length: 每个样本的长度
        :param lstm_num_units: lstm节点中隐层节点数目
        :param keep_prob: 随机drop比例，防止过拟合
        :param num_layers: lstm节点层数
        :param learning_rate: 学习率
        :param clip_val: 梯度最大值，防止梯度爆炸，如果梯度过大，则依据此值进行裁剪
        '''

        #初始化模型的input和target
        self.input, self.target = model_input(n_seqs=n_seqs, n_sequencd_length=n_sequencd_length)
        #构建lstm
        cell, self.init_state = model_lstm(lstm_num_units=lstm_num_units, keep_prob=keep_prob, num_layers=num_layers,
                                      n_seqs=n_seqs)
        #目前输入的内容是编码后的文本，是一个个数字表示，这里把它转换成onehot向量形式
        input_one_hot = tf.one_hot(self.input, len(vocab))

        #运次lstm
        outputs, self.state = tf.nn.dynamic_rnn(cell, input_one_hot, initial_state=self.init_state)

        #计算该batch的输出
        self.predtion, logits = model_output(lstm_output=outputs, in_size=lstm_num_units, out_size=len(vocab))

        #计算损失
        self.loss = model_loss(target=self.target, logits = logits, num_class=len(vocab))
        #梯度裁剪
        self.optimizer = model_optimizer(learning_rate=learning_rate, loss=self.loss, clip_val=clip_val)

n_seqs=100
n_sequencd_length=100
lstm_num_units=512
num_layers=2
learning_rate=0.01
keep_prob=0.5

if __name__ == '__main__':
    '''
    整体步骤分为三部分：
    1.数据预处理：
        数据加载、数据预处理、数据切片batch
    2.模型构建
            输入层、lstm层、输出层、loss、optimizater
    3.训练
    4.生成文本
    '''
    #加载数据
    vocab, vocab2Int, int2Vocab, encode = loadData('data/Harry_Potter1-7.txt')
    #初始化模型
    char_rnn = char_RNN(vocab=vocab, n_seqs = n_seqs, n_sequencd_length = n_sequencd_length,
                        lstm_num_units=lstm_num_units, keep_prob=keep_prob, num_layers=num_layers,
                 learning_rate=learning_rate, clip_val=5)

    saver = tf.train.Saver()

    #设置迭代轮数
    epochs = 200
    #全局计数
    count = 0

    with tf.Session() as sess:
        #初始化所有变量
        sess.run(tf.global_variables_initializer())

        #进行轮数迭代
        for epoch in range(epochs):
            #每次获取一个batch，进行训练
            for x, y in get_batch(input_data=encode, n_seqs=n_seqs, n_sequencd_length=n_sequencd_length):
                count += 1

                feed = {
                    char_rnn.input:x,
                    char_rnn.target:y
                }

                #训练
                _, loss, _ = sess.run([char_rnn.state, char_rnn.loss, char_rnn.optimizer], feed_dict=feed)

                #定期打印数据
                if count % 500 == 0:
                    print('-----------------------------')
                    print('轮数：%d:%d' % (epoch + 1, epochs))
                    print('训练步数：%d' % (count))
                    print('训练误差:%.4f' % (loss))
            #定期保存ckpt
            if epoch % 5 == 0:
                saver.save(sess, 'checkpoint/model.ckpt',global_step=count)

        saver.save(sess, 'checkpoint/model.ckpt', global_step=count)
