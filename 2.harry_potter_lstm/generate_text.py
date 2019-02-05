from harry_potter_lstm import *

def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符

    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_num_units, vocab_size, prime="The "):
    '''
    生成文本
    :param checkpoint:
    :param n_samples:
    :param lstm_num_units:lstm内部隐层节点数目
    :param vocab_size: 词汇表大小
    :param prime: 开头字母
    :return:
    '''

    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    #创建模型
    model = char_RNN(vocab=vocab, n_seqs = 1, n_sequencd_length = 1,
                    lstm_num_units=lstm_num_units, keep_prob=1, num_layers=num_layers,
                 learning_rate=learning_rate, clip_val=5)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        #生成初始状态
        new_state = sess.run(model.init_state)
        #获取下一个字符
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = vocab2Int[c]
            feed = {model.input: x,
                    model.init_state: new_state}
            preds, new_state = sess.run([model.predtion, model.state],
                                        feed_dict=feed)
        c = pick_top_n(preds, len(vocab))
        # 添加字符到samples中
        samples.append(int2Vocab[c])

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.input: x,
                    model.init_state: new_state}
            preds, new_state = sess.run([model.predtion, model.state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int2Vocab[c])

    return ''.join(samples)



if __name__ == '__main__':
    #加载文件
    vocab, vocab2Int, int2Vocab, encode = loadData('data/Harry_Potter1-7.txt')
    #读取checkpoint
    checkpoint = tf.train.latest_checkpoint('checkpoint/')
    print(checkpoint)

    #生成文本
    samp = sample(checkpoint, 150, lstm_num_units, len(vocab), prime="Hi, ")
    print('--------------------------------')
    print(samp)
