#coding=utf-8
#Author:Dodo
#Date:2019-01-29
#Blog:www.pkudodo.com

'''
# 文件结构:
#   skip_gram.py:训练词嵌入矩阵
#
# 训练情况：
#   数据集：维基百科文本
#   已训练step:524500
#
# 训练结果：
#   随机抽取16个词，根据其词向量找到最接近的8个词
# Nearest to [cahokia]: luanda, platte, johannesburg, sacking, grotto, seleucids, saronic, ramparts,
# Nearest to [nsync]: flamsteed, cru, strapped, aspiring, lesh, altenberg, legitimized, reclining,
# Nearest to [carlin]: keane, oliphant, comedienne, conservationist, maloney, ballets, baresi, showtime,
# Nearest to [operatorname]: varphi, equiv, exp, bigg, widehat, cdots, langle, sqrt,
# Nearest to [amin]: karzai, mansur, ahmad, overthrown, bloodless, kabila, weimar, reformist,
# Nearest to [hemoglobin]: ligand, molecules, ligands, photosynthesis, aerobic, enzyme, pancreatic, chlorophyll,
# Nearest to [goin]: allyn, brenda, undertones, freeform, tramp, boomers, brewer, lamont,
# Nearest to [financi]: communaut, mise, publique, sie, republica, rcito, centimes, compostela,
# Nearest to [caesarion]: childless, tiberius, caracalla, begotten, bothwell, desiderius, clemency, quarrelled,
# Nearest to [inserted]: differentiating, cinnamon, spindle, cds, disks, tentacle, mrna, genie,
# Nearest to [romano]: musketeers, oro, groningen, canaria, bernardino, madame, novellas, yoruba,
# Nearest to [simplest]: cooh, nucleophilic, iterated, alkyl, silicates, assembler, organic, acetylene,
# Nearest to [stipulate]: deluded, skookum, hypocrites, sanctification, remarry, mashiach, persuades, dervish,
# Nearest to [automorphisms]: quotient, injective, adjoint, isomorphic, bilinear, automorphism, homomorphism, homeomorphic,
# Nearest to [penguin]: paperback, hardcover, heaney, lily, bujold, toaster, brendan, prometheus,
# Nearest to [trinidad]: tobago, kitts, suriname, grenadines, tuvalu, swaziland, guyana, tonga,
'''
import string
from collections import Counter
import math
import random
import tensorflow as tf
import numpy as np
import time

def loadData(fileName, low_threshold = 10, high_freq_threshold = 0.85):
    #加载文件
    text = open(fileName).read().lower()

    #删除标点符号，并以空格切分字符串，形成单个单词
    #punctuation内部是ascii中包含的一些标点符号
    for c in string.punctuation:    # ， . : ; !
        text = text.replace(c, ' ')
    #切分
    text = text.split()

    #删除频率小于10的低频次
    #过于低频的词会造成一些干扰
    wordCount = Counter(text)
    text = [word for word in text if wordCount[word] >= low_threshold]

    #删除高频词，比如说the, a , of等等无意义的高频词
    #这里删除高频词用的公式是：
    #   p(w) = 1 - sqrt(t / f(w))
    #   p(w)是一个词出现的频率(严格来讲其实不是，暂时先这么认为)
    #   f(w)是当前word的频率，f(w) = count(w) / count(text)，频率是频数除以文本包含的总词数
    #   t是一个设定的值，一般在1e-5到1e-3之间
    #   然后认为p(w) > 8的单词为高频词，对其删除掉
    #-------------------------------------------------
    #   上面的计算公式是在知乎上看到的，但是感觉实际上并没有多大意义，它仍然表示的是一个单词频率达到
    #   一定次数就删除。
    #   比如说如果文本长度时1w， p(w)>0.8为高频词的话，
    #   按照公式反向计算，出现250次以上就认为是高频词了，也就是说，不需要进行公式计算，直接对25进行截断就好了
    high_freq = 1e-3
    wordCount = Counter(text)
    totalCount = len(text)
    word_frep = {word: (1 - math.sqrt(high_freq / (count / totalCount))) for word, count in wordCount.items()}
    #保留非高频词
    text = [word for word in text if word_frep[word] < high_freq_threshold]

    #创建字典
    vocab = set(text)
    vocab_List = list(vocab)
    vocab_List.sort()

    #创建两种索引的词典
    word2Int = {word:index for index, word in enumerate(vocab_List)}
    int2Word = {index: word for word, index in word2Int.items()}

    #对文本进行编码
    encode = [word2Int[word] for word in text]

    return vocab, word2Int, int2Word, encode

def get_windows_word(text, word_id, window_size):
    '''
    找到指定单词所在窗口内的其他单词
    :param text: 文本
    :param word_id: 单词所有
    :param window_size: 窗口大小
    :return:
    '''
    #找到窗口的起始位置
    start = word_id - window_size if word_id - window_size >=0 else 0
    #找到窗口的终点位置
    end = word_id + window_size if word_id + window_size < len(text) else (len(text) - 1)
    #得到窗口内除给定单词外的其他单词
    window_word = set(text[start: word_id] + text[word_id + 1: end + 1])
    return list(window_word)

def get_batch(text, windows_size, batch_size):
    '''
    返回生成器，迭代生成batch
    :param text:文本
    :param windows_size:窗口尺寸
    :param batch_size:batch尺寸
    :return:
    我喜欢春天的美丽和温暖

    size = 3
    中心词：我
    周围词：喜欢春
    训练集：
        （我， 喜）
        （我， 欢）
        （我， 春）
    '''
    x, y = [], []
    for id in range(len(text)):
        #对每个单词，找到它窗口内的其他单词
        window_word = get_windows_word(text, id, windows_size)
        #生成训练样本（单词，目标单词）
        #窗口中每个单词都对应生成一个训练集，比如说给定单词是x，窗口中其他单词有10个，那么会生成10个样本
        #x中需要添加len(window_word个重复项，以此保证x和y的匹配
        x.extend([text[id]] * len(window_word))
        y.extend(window_word)

    #打乱训练集
    #这步其他地方没见到过有要求，我在写的时候感觉应该将训练集打乱，
    #但是我提供不了理论依据，只是一种感觉，所以为了以防万一，我就写上了
    #你如果觉得没有必要，下面三行删掉也没有问题的
    combine = list(zip(x, y))
    random.shuffle(combine)
    x[:], y[:] = zip(*combine)

    #将训练文本变成可以整除batch_size
    n_batch = len(text) // batch_size
    text = text[: n_batch * batch_size]

    #迭代获得所有batch
    for i in range(0, len(text), batch_size):
        batch_x = x[i: i + batch_size]
        batch_y = y[i: i + batch_size]

        yield  batch_x, batch_y

def model_build(vocab_size, embedding_size, n_sample):
    '''
    skip-gram  cbow
    skip-gram   中心词     -》    周围词
    cbow        周围词     -》    中心词

    我喜欢足球       喜欢   ->   名词：我  你  他    足球  保龄球   写程序

    我爱好足球       爱好  -》     名词：我  你  他    足球  保龄球   写程序


    创建模型结构

    整个结构分三层
    1.输入层：
    2.隐层
    3.softmax

    输入层->隐层：
        先理解成输入x是onehot向量，在输入层和隐层之间有一个权值矩阵w，如果词汇表大小是1w，隐层设置成300（一般在200-500）
        之间。那么输入层和隐层之间的矩阵w大小为(1w,300)，实际上训练结束后每一行代表一个word，横向的300维是这个word的特征，
        所以说这个w实际上就是我们训练结束后得到的嵌入词向量。每一行都是一个词的向量。

    通俗点先这么理解，整个流程是这样的：我们有一个大矩阵，每一行代表一个词，我们要学的就是每个词自身的那300维特征，如果
    词相似，那么特征也应该相似。
    skip-gram是给定一个词，去预测边上的词，那么在训练过程中，我们先找到这个词指向的300维特征，特征有了，已经能锁定这个词的
    特性了，这个时候直接用一层神经网络输出，再通过softmax，得到哪些单词的概率最高。

    三层的功能就是这么规划的：
    1.输入层输入单词的onehot向量
    2.onehot与w相乘，是[1, 10000] * [1000, 300] = [1, 300]的向量，其实得到的就是这个词所指向的那300维特征
    3.隐层得到了300维特征，再通过一个wx+b， 之后softmax，得到概率输出
    注意：第二步的乘法中，onehot只有一个1，其他全是0，是无意义的计算，所以使用了embedding_lookup优化计算，
    本质上就是直接查找该300维特征来替换掉乘法，提高速度

    :param vocab_size:词汇表大小
    :param embedding_size: 嵌入层尺寸
    :param n_sample:每批随机抽样的类数
    :return:
    '''

    #输入和标签
    # input = tf.placeholder(dtype=tf.int32, shape=(vocab_size), name='input')
    # label = tf.placeholder(dtype=tf.int32, shape=(vocab_size, None), name='label')
    input = tf.placeholder(dtype=tf.int32, shape=(None), name='input')
    label = tf.placeholder(dtype=tf.int32, shape=(None, None), name='label')

    #获取词嵌入向量
    embedding = tf.Variable(tf.truncated_normal(shape=(vocab_size, embedding_size), stddev=0.1), name='embedding')
    embed = tf.nn.embedding_lookup(params=embedding, ids=input, name='embed')

    #输出层
    #softmax_w的维度，其实应该是(embedding_size, vocab_size)，但是在代码中倒过来了，在sampled_softmax_loss
    #源码中有些详细原因，主要是该函数对参数维度的要求就是这样的，下面这句话是从tf.nn.sampled_softmax_loss源码注释
    #中复制过来的
    #------------------------------------------------------------------------------------------------------
    # weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor` objects whose concatenation
    # along dimension 0 has shape [num_classes, dim].The(possibly - sharded) class embeddings.
    #------------------------------------------------------------------------------------------------------
    softmax_w = tf.Variable(tf.truncated_normal(shape=(vocab_size, embedding_size), stddev=0.1), name='softmax_w')
    softmax_b = tf.Variable(tf.zeros(vocab_size), name='softmax_b')

    #损失
    loss = tf.nn.sampled_softmax_loss(weights=softmax_w, biases=softmax_b, labels=label, inputs=embed,
                                      num_sampled=n_sample, num_classes=vocab_size)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)


    #下面部分代码是为了在训练过程中可以看到单词的联系
    #在词汇表中随机挑选16个单词
    valid_size = 16
    valid_examples = np.array(random.sample(range(vocab_size), valid_size))
    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

    return input, label, loss, optimizer, valid_size, similarity, valid_examples, normalized_embedding


def model_train(text, vocab_size, int_to_vocab, epochs = 20, embedding_size=300, n_sample=100,
                windows_size = 5, batch_size = 100, ):
    '''
    模型训练
    :param text:编码后的文本
    :param vocab_size: 所有字符的集合的长度
    :param int_to_vocab: 索引->word的字典
    :param epochs: 迭代器轮数
    :param embedding_size: 嵌入层大小
    :param n_sample:交叉熵中的参数：每批随机抽样的类数
    :param windows_size: 窗口尺寸
    :param batch_size: 每个batch大小
    :return:
    '''
    #初始化模型
    input, label, loss, optimizer, valid_size, similarity, valid_examples, normalized_embedding\
        = model_build(vocab_size, embedding_size, n_sample)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        count = 0
        for epoch in range(epochs):
            #获取每个batch
            for x, y in get_batch(text, windows_size, batch_size):
                start = time.time()

                count += 1
                feed = {
                    input:x,
                    label:np.array(y)[:, None]
                }
                #训练
                batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed)
                end = time.time()

                #定期打印数据
                if count % 300 == 0:
                    print('epoch:%d/%d'%(epoch, epochs))
                    print('count:', count)
                    print('time span:', end - start)
                    print('loss:', batch_loss)

                #定期打印文本学习情况
                if count % 500 == 0:
                    # 计算similarity
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = int_to_vocab[valid_examples[i]]
                        top_k = 8  # 取最相似单词的前8个
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to [%s]:' % valid_word
                        for k in range(top_k):
                            close_word = int_to_vocab[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)

                #定期保存模型
                if count % 500 == 0:
                    saver.save(sess, "checkpoints/model.ckpt", global_step=count)
                    embed_mat = sess.run(normalized_embedding)
                    print('----------------')
                    print(type(embed_mat))
                    print('- - - - - - - - ')
                    print(embed_mat)
                    print('- - - - - - - - ')

epochs = 10
embedding_size = 300
windows_size = 5
batch_size = 100
if __name__ == '__main__':
    #加载文件
    vocab, word2Int, int2Word, encode = loadData('data/text')
    #模型训练
    model_train(encode, len(vocab), int2Word, epochs=epochs, embedding_size=embedding_size,
                windows_size=windows_size, batch_size=batch_size)
