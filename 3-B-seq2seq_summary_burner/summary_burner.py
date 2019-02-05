#coding=utf-8
#Author:Dodo
#Date:2019-02-02
#Blog:www.pkudodo.com

'''
文件结构:
  summary_burner.py:训练模型
  general_summary.py:摘要生成测试

训练结果：
------------the text is:----------------
Use olive oil to cook this, salt it well, and it is the best, most tender popcorn I have ever eaten. I add a tiny bit of butter to mine, but don't need it. My nine year old daughter didn't like popcorn until she reluctantly tried this. After a few bites, she consumed half the bowl!<br /><br />I bought mine at a specialty popcorn shop in Long Grove IL, so I didn't have to pay shipping costs, but when it's gone, I might have to bite the bullet and order it from here.",Spoiled me for other popcorn
------------the summary is:-------------
best tasting popcorn ever


------------the text is:----------------
I bought this with the whirly pop popcorn popper, and I made some as soon as I got them. It is the best popcorn I've had. I thought I would mind the smaller kernel size, but I didn't. There is substantially less hull than regular popcorn. The flavor is great. I suggest using peanut oil for popping popcorn. It gives popcorn a great nutty taste - really delicious. You can buy peanut oil at any oriental (Asian) store. It's better than olive oil (olive oil and popcorn is too weird for me). But peanut oil is perfect. You won't need butter flavor or anything like that again (artificial butter flavorings are full of chemicals). Now, we pop around three times a day and always crave this for a snack instead of junkfood. Try it and you'll be popping like us.
------------the summary is:-------------
great taste
'''

import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

import time

def load_data(fileName):
    #载入文件
    df = pd.read_csv(fileName, encoding='utf-8')
    #获取需要的两列
    df = df[['Text', 'Summary']]
    #删除值为空的行
    df = df.dropna()
    #重新建立索引
    df = df.reset_index(drop=True)
    #获取5w个样本作为训练数据
    df = df[:50000]
    #释放一些空间
    return df

def clean_data(df, isSummary=False):
    all_text = []
    #载入停用词
    en_stops = set(stopwords.words('english'))
    #加载评论文本
    if isSummary == False:
        text = df['Text']
    else:
        text = df['Summary']
    for line in text:
        #转换成小写
        line = line.lower()
        #去标点符号
        for c in string.punctuation:
            line = line.replace(c, ' ')
        #分词
        words = line.split(' ')
        #如果不是总结，则删除停用词
        if isSummary == False:
            words = [word for word in words if word not in en_stops]
        all_text.append(words)
    return all_text

def get_embedings(text, summary, embeding_fileName):
    '''
    创建本地词嵌入矩阵
    :param text:文本
    :param summary: 总结
    :param embeding_fileName:原始词向量文件路径
    :return:
        cur_vocab2int：字典， 单词 -> 索引
        cur_int2vocab：字典： 索引 -> 单词
        cur_embedings：本地生成的词嵌入矩阵
        text：编码后的text
        summary：编码后的summary
    '''
    '''
    原始的词向量来源于github：https://github.com/commonsense/conceptnet-numberbatch
    下载地址：https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.04b.txt.gz
    为txt格式，每行是开头是一个单词，后面是300维该单词的向量
    例如：
        a -0.0248 0.2166 -0.0618 -0.0334 -0.0008 0.0507 .......
    
    思路：
        1.训练文本处理， 步骤可以参考clean_data()
        2.统计文本中所有出现过的词，建立本地词汇表
        3.根据文本中出现过的词去原始词向量中查询对应向量
        4.建立本地词嵌入矩阵：
            建立本地词矩阵主要原因：可以建立索引，后续步骤中能有方法查到对应词的词向量
                在生成本地词汇表的过程中，生成了字典cur_vocab2int，其中键为单词，值为单词索引
                同时建立本地词嵌入矩阵cur_embedings，其中对应索引即为该单词词向量
                获取方式可以是这样：cur_embedings[cur_vocab2int['hello']]。
                如果不这么处理，没有办法很快地根据指定单词，在原始词向量中获取到指定向量                
    '''
    #原始矩阵字典，原始词嵌入矩阵中所包含的所有词和对于向量
    #其中键为单词，值为该单词的300维向量
    ori_embedings_dict = {}

    #载入原始词嵌入矩阵
    ori_embedings_file = open(embeding_fileName, encoding='utf-8')
    for line in ori_embedings_file.readlines():
        #读取每一行，按照空格切分元素
        words = line.split(' ')
        #每行的第一个元素是单词，后续元素是该单词的向量
        #存入字典中便于后面查询
        ori_embedings_dict[words[0]] = words[1:]

    #特殊字符
    #<GO>:一句话的开头：在后续步骤中需要手动给每句话开头加上<GO>，具体原因后面再写
    #<EOS>：一句话的结尾：与go一样，需要标明EOS。同样，当预测的时候，如果预测结果为EOS，说明一句话结束，也就停止预测
    #<PAD>：填充字符：在每个batch内的训练样本长度需要一致，所以如果长度未够的，需要填充
    #<UNK>：未知字符：没有在词汇表中出现过的单词，用UNK表示
    special_words = ['<GO>', '<EOS>', '<PAD>', '<UNK>']
    #遍历评论和总结样本，获取到里面所有出现过的单词，将单词放入集合中去重
    #最后再list处理，是为了后续能够排序。
    #排序是因为获得所有单词以后，要建立vocab2int和int2vocab两个索引字典，但是在下面这一行建立set的过程中，其实单词
    #在set中存放的顺序是不一定的，这样会导致生成的缩印字典每次运行也都不一致。如果程序不结束，训练完以后直接预测，是没有
    #问题的，但如果分开运行，在预测过程中需要重新生成字典，生成的字典基本上是不会一致的，所以没有办法使用。
    #这里排序主要就是因为这个，这样能保证在list中，每次运行后，所有单词在内部排序是一样的，后续生成的字典也是一样的
    cur_vocab_list = list(set([word for line in (text + summary) for word in line])) + special_words
    cur_vocab_list.sort()

    #建立两个字典
    #cur_vocab2int：  词汇 -> 索引       用于文本处理中的编码工作，获得索引后在词嵌入矩阵中获得对应向量
    #cur_int2vocab：  索引 -> 词汇       在预测过程中，输出的编码需要转换到对应的单词
    cur_vocab2int = {}
    cur_int2vocab = {}
    #根据所有出现过的词，建立字典
    for index, word in enumerate(cur_vocab_list):
        cur_vocab2int[word] = index
    for word, index in cur_vocab2int.items():
        cur_int2vocab[index] = word

    #初始化词嵌入矩阵， 大小为 词汇量 * 300
    cur_embedings = np.zeros((len(cur_vocab2int), 300), dtype=np.float32)
    #遍历所有出现过的词
    for word in cur_vocab_list:
        #如果在原始词嵌入矩阵中能够找到
        if word in ori_embedings_dict.keys():
            #从原始词嵌入字典中找到其对应的向量，放入本地词嵌入矩阵中对应索引中
            cur_embedings[cur_vocab2int[word]] = np.array(ori_embedings_dict[word])
        else:
            #如果该单词在原始词嵌入矩阵中也没有，那么就随机初始化一个向量
            #它会在训练过程中进行学习的
            cur_embedings[cur_vocab2int[word]] = np.random.uniform(-1, 1, 300)

    #将所有文本从单词形式转换为编码形式
    text = [[cur_vocab2int[word] for word in line] for line in text]
    #总结文本转换为编码形式
    summary = [[cur_vocab2int[word] for word in line] + [cur_vocab2int['<EOS>']] for line in summary]

    #在模型中使用的是动态lstm，静态lstm的结构是提前展开的，比如说设置10个lstm单元排成一排，那么文本大小就必须为10，
    #动态lstm只有一个lstm单元，它会不断循环该lstm单元，等于说重复利用，所以长度可以不一致。但是在同一个batch内，所有样本
    #长度需保持一致。
    #所以这里先根据text的长度从小到大进行排序，比如说一个batch_size是5， 排序后前5个text长度分别是3， 3， 3， 4， 5。那么
    #在该batch的训练中，所有样本只需要填充到长度5就可以了。如果不排序，前5个可能是3， 6， 10， 2， 1。为了统一，所以样本需要填充
    #到10，很浪费资源，效果也不好。
    combine = [[len(text[i]), text[i], summary[i]]for i in range(len(text))]
    combine = sorted(combine, key=lambda x: x[0], reverse=False)
    for i in range(len(combine)):
        text[i] = combine[i][1]
        summary[i] = combine[i][2]
    return cur_vocab2int, cur_int2vocab, cur_embedings, text, summary

def seq2seq_get_input():
    '''
    创建模型输入
    :return:
    '''
    #text输入，shape=[None, None],第一个None可以写成batch_size，这里就直接None了，第二个None是因为不同的
    #batch大小不一样，所以没有办法给定一个确定数
    text_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='text_input')
    #在训练过程中的target，也就是summary， 与text_input应该保持一致
    summary_output = tf.placeholder(dtype=tf.int32, shape=[None, None], name='summary_output')

    #文本长度，是一个列向量，每一个元素是该行text的长度
    text_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='text_sequence_length')
    #总结文本长度，也是列向量，每个元素是该行summary长度
    summary_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='summary_sequence_length')
    #在该batch内总结文本的最大长度， 后续要用到
    summary_sequence_max_length = tf.reduce_max(summary_sequence_length)

    return text_input, summary_output, text_sequence_length, summary_sequence_length,summary_sequence_max_length

def seq2seq_create_rnn_cell(rnn_size, keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
    # cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
    return cell

def seq2seq_create_encoder(embeddings, text_input, keep_prob, num_layers, rnn_size, text_sequence_length):
    '''
    创建模型的encoder
    :param embeddings:词嵌入矩阵
    :param text_input: 输入文本
    :param keep_prob:
    :param num_layers: lstm的层数
    :param rnn_size: rnn内部节点数量
    :param text_sequence_length: 文本长度
    :return:
    '''
    #获取到输入的文本对应的词嵌入向量
    #具体可以百度tf.nn.embedding_lookup
    text_embed = tf.nn.embedding_lookup(embeddings, text_input)

    enc_output, enc_final_state = None,None
    #对每一层构建前向和反向rnn
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            #前向rnn
            cell_fw = seq2seq_create_rnn_cell(rnn_size, keep_prob)
            #反向rnn
            cell_bw = seq2seq_create_rnn_cell(rnn_size, keep_prob)
            #创建
            enc_output, enc_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, text_embed, text_sequence_length, dtype=tf.float32)

    #双向rnn
    encoder_output = tf.concat(enc_output, 2)

    return encoder_output, enc_final_state


def seq2seq_create_decoder(enc_output,
                           encoder_final_state,
                           text_sequence_length,
                           summary_sequence_length,
                           summary_sequence_max_length,
                           embeddings,
                           decode_input,
                           keep_prob,
                           num_layers,
                           rnn_size,
                           batch_size,
                           vocab2int
                           ):
    '''
    模型decode层
    :param enc_output:encode的输出
    :param encoder_final_state:encode输出的状态
    :param text_sequence_length:文本长度
    :param summary_sequence_length:总结长度
    :param summary_sequence_max_length:总结的最大长度
    :param embeddings:词嵌入矩阵
    :param decode_input:decode的输入
    :param keep_prob:
    :param num_layers:lstm层数
    :param rnn_size:rnn内部节点数量
    :param batch_size:
    :param vocab2int:
    :return:
    '''
    #获取总结文本的词嵌入
    summary_embed = tf.nn.embedding_lookup(embeddings, decode_input)

    def get_rnn_cell(rnn_size):
        '''
        获取单层lstm
        :param rnn_size:
        :return:
        '''
        #添加注意力机制
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size, enc_output, text_sequence_length, normalize=False)
        #创建单层lstm
        cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
        #将注意力机制与lstm绑定
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism=attn_mech, attention_layer_size=rnn_size)
        return cell

    #多层lstm
    cell = tf.contrib.rnn.MultiRNNCell([get_rnn_cell(rnn_size) for _ in range(num_layers)])
    #lstm输出后，用一个全连接层来输出表示的是哪个word
    output_layer = Dense(len(vocab2int), kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

    #这里用了两种decode模式
    #一种用于训练，一种用于预测
    #decode中的每一个lstm节点都会输出一个word， 同时作为下一个节点的一个输入
    #那么在训练过程中，刚开始训练的时候，节点输出的结果不一定是正确的，一个错误的结果传到下一个节点，输出当然大概率也不会
    #正确，所以这样子进行训练，只有前面的节点预测结果正确，后面的才能进行有效训练。这样子训练效率太慢了，效果也不好。
    #所以在训练过程中，每个节点有一个预测输出，但是它并不作为下一个节点的输入，而是从summary_embed中得到当前及节点
    #应该输出的正确值，也就是标记，将该标记作为下一个节点的输入，这样使得所有节点在都能得到正确的训练

    #下面两块第一块是训练的decode， 第二个是预测的decode， 他们使用同一个命名空间，主要是因为虽然是两块，但是我们希望
    #两个decode内部的权重矩阵是一致的，也就是说，训练得到的参数，在预测过程中能用上，两个是共享的。
    #关于两个decode内部用的api区别，自行百度一下， 主要区别就在于TrainingHelper和GreedyEmbeddingHelper，
    #两个函数分别设置了两种模式，TrainingHelper是前一个输出不作为下一个输入，而是使用标记作为输入
    #GreedyEmbeddingHelper是前一个输出作为下一个输入

    #训练
    with tf.variable_scope('decode'):
        #创建helper， 对decode模式进行设置
        train_helper = tf.contrib.seq2seq.TrainingHelper(summary_embed, summary_sequence_length, time_major=False)
        #创建decode
        train_basic_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, cell.zero_state(dtype=tf.float32,batch_size=batch_size), output_layer)
        #运次
        train_decoder_outputs, _, _ = \
            tf.contrib.seq2seq.dynamic_decode(train_basic_decoder, impute_finished=True, maximum_iterations=summary_sequence_max_length)

    with tf.variable_scope('decode'):
        #在预测过程中，需要手动给每一个文本添加GO标签
        start_tokens = tf.tile(tf.constant([vocab2int['<GO>']], dtype=tf.int32), [batch_size])
        #创建helper， 设置前一个输出作为后一个输入
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens, vocab2int['<EOS>'])
        #创建deocde
        pred_basic_decoder = tf.contrib.seq2seq.BasicDecoder(cell, pred_helper, cell.zero_state(dtype=tf.float32,batch_size=batch_size), output_layer)
        #运行
        pred_decoder_outputs, _, _ = \
            tf.contrib.seq2seq.dynamic_decode(pred_basic_decoder, impute_finished=True, maximum_iterations=summary_sequence_max_length)

    return train_decoder_outputs, pred_decoder_outputs

def create_seq2seq_model(embedings, keep_prob, num_layers, rnn_size, batch_size, learn_rate, vocab2int):
    '''
    创建seq2seq模型
    :param embedings:词嵌入矩阵
    :param keep_prob:
    :param num_layers: lstm层数
    :param rnn_size: rnn内部隐层节点数量
    :param batch_size: 一个batch大小
    :param learn_rate: 学习率
    :param vocab2int: 字典   单词 -> 索引
    :return:
    '''
    #创建tensor
    text_input, summary_target, text_sequence_length, summary_sequence_length, summary_sequence_max_length\
        = seq2seq_get_input()
    #创建encoder
    encoder_outputs, encoder_final_state = \
        seq2seq_create_encoder(embeddings=embedings, text_input=text_input, keep_prob=keep_prob, num_layers=num_layers,
                               rnn_size=rnn_size, text_sequence_length=text_sequence_length)

    #给每一个总结性文本都去掉最后一个word，同时在开头添加<GO>
    #最屌最后一个word是因为:
    #   例如有文本是下面这样：
    #       i       like        it      <EOS>   <PAD>   <PAD>   <PAD>
    #       i       think       it      is      not     good    <EOS>
    #       my      mother      like    it      <EOS>   <PAD>   <PAD>
    #在lstm的decode端中，每一个预测输出的单词，都是下一个节点的输入，但是当预测到最后一个单词时，
    #后面没有节点了， 所以不会再变成下一个节点的输入，所以也就不需要了
    #这里感觉用文字讲不太清，可以看我的b站讲解视频，我对这块有详细说明
    ending = tf.strided_slice(summary_target, [0, 0], [batch_size, -1], [1, 1])
    decode_input = tf.concat([tf.fill([batch_size, 1], vocab2int['<GO>']), ending], 1)
    #创建decode
    train_decoder_outputs, pred_decoder_outputs = seq2seq_create_decoder(encoder_outputs, encoder_final_state, text_sequence_length, summary_sequence_length,
                                                                         summary_sequence_max_length, embedings,
                                                                         decode_input, keep_prob, num_layers,
                                                                         rnn_size, batch_size, vocab2int)
    #复制tensor
    #关于rnn_output和sample_id属性：
    #   rnn_output是decode输出以后，通过全连接层输出的向量，这里用来计算损失
    #   sample_id可以理解成全连接层输出的向量，通过softmax后计算出来的最有可能的word索引
    train_logits = tf.identity(train_decoder_outputs.rnn_output, name='train_logirs')
    predict = tf.identity(pred_decoder_outputs.sample_id, name='predict')
    #mask函数可以百度一下，看看实例。
    #主要作用是在因为summary为了填充到一致，里面有很多的<pad>， 实际上没有什么意义。在计算损失的时候，
    #要把这些pad排除在外。
    mask = tf.sequence_mask(summary_sequence_length, summary_sequence_max_length, dtype=tf.float32)#################
    with tf.variable_scope('optimizer'):
        #极速前损失
        loss = tf.contrib.seq2seq.sequence_loss(train_logits, summary_target, mask, )
        #设置学习率
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        #梯度裁剪，防止梯度爆炸
        #loss对遍历进行求导
        gards = optimizer.compute_gradients(loss)
        #遍历所有梯度，对梯度进行裁剪
        gards = [(tf.clip_by_value(gard, -5, 5), val) for gard, val in gards if gard is not None]
        #更新修改后的梯度
        train_op = optimizer.apply_gradients(gards)

    return text_input, summary_target, text_sequence_length, summary_sequence_length, loss, train_op


def get_batch(input_data, target, batch_size, vocab2int):
    '''
    获取一个batch
    :param input_data:输入文本
    :param target: summary
    :param batch_size: 一个batch的大小
    :param vocab2int:
    :return:
    '''
    #使用batchsize对文本长度进行整除，查看有多少个batch，最后不能整除的部分就扔掉了
    n_batch = len(input_data) // batch_size

    for i in range(0, (n_batch * batch_size - 1), batch_size):
        #获得当前batch
        batch_input = input_data[i: i + batch_size]
        batch_target = target[i: i + batch_size]

        input_length = []
        # 获得batch内每一个text的长度
        for sample in batch_input:
            input_length.append(len(sample))

        target_length = []
        # 获得batch内每一个summary的长度
        for sample in batch_target:
           target_length.append(len(sample))

        #计算batch内最大的文本长度
        input_max_length = max(input_length)
        #将所有文本填充到同一最大长度
        batch_input = np.array([sample + [vocab2int['<PAD>']] * (input_max_length - len(sample)) for sample in batch_input], dtype=np.int32)

        #计算batch内最大的总结长度
        target_max_length = max(target_length)
        #将所有总结填充到同一最大长度
        batch_target = np.array([sample + [vocab2int['<PAD>']] * (target_max_length - len(sample)) for sample in batch_target], dtype=np.int32)

        #构造生成器，迭代返回batch
        yield batch_input, batch_target, input_length, target_length


epoches = 6        #迭代次数
keep_prob = 0.8
num_layers = 2      #lstm层数
rnn_size = 256      #rnn内部隐层节点数量
batch_size = 64     #一个batch大小
learn_rate = 0.005   #学习率
checkpoint = 'checkpoint/model'
save_num = 20       #每save_num个batch保存一次模型

if __name__ == '__main__':
    #加载数据
    df = load_data('data/Reviews.csv')
    #清洗数据
    text = clean_data(df, isSummary=False)
    summary = clean_data(df, isSummary=True)
    #建立词嵌入矩阵
    vocab2int, int2vocab, embedings, text, summary = get_embedings(text, summary, 'data/numberbatch-en-17.04b.txt')
    #建立模型
    text_input, summary_target, text_sequence_length, summary_sequence_length, loss, train_op = \
        create_seq2seq_model(embedings, keep_prob, num_layers, rnn_size, batch_size, learn_rate, vocab2int)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #初始化所有变量
        sess.run(tf.global_variables_initializer())
        count = 0
        #迭代
        for epoch in range(epoches):
            #获得一个batch
            for batch_text, batch_summary, batch_text_length, batch_summary_length in get_batch(text, summary, batch_size, vocab2int):
                count += 1

                start = time.time()
                print('start a batch:', count)
                feed = {
                    text_input:batch_text,
                    summary_target:batch_summary,
                    text_sequence_length:batch_text_length,
                    summary_sequence_length:batch_summary_length
                }
                #获得损失
                batch_loss, _ = sess.run([loss, train_op], feed_dict=feed)
                #打印结果
                if count % 1 == 0:
                    print('epoch:%d/%d, count:%d, loss:%.4f'%(epoch, epoches, count, batch_loss))
                #保存模型
                if count % save_num == 0:
                    saver.save(sess, checkpoint)
                print('time span:', time.time() - start)
        saver.save(sess, checkpoint)


