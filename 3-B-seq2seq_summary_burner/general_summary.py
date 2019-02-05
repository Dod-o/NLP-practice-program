from summary_burner import *

def test_clean_text(text):
    '''
    清洗数据
    :param text:
    :return:
    '''
    #载入英文中的停用词  类似  a  the  this.....
    en_stops = set(stopwords.words('english'))
    #全部切换成小写
    text = text.lower()
    #删除标点符号
    for c in string.punctuation:
        text = text.replace(c, '')
    text = text.split()
    #删除停用词
    text = [word for word in text if word not in en_stops]
    return  text

def get_vocab_dict():
    df = load_data('data/Reviews.csv')
    text = clean_data(df, isSummary=False)
    summary = clean_data(df, isSummary=True)
    vocab2int, int2vocab, _, _, _ = get_embedings(text, summary, 'data/numberbatch-en-17.04b.txt')

    return vocab2int, int2vocab

if __name__ == '__main__':
    #text文本
    # ori_test_text = 'This taffy is so good.  It is very soft and chewy.  The flavors are amazing.  I would definitely recommend you buying it.  Very satisfying!!'
    # ori_test_text = 'Use olive oil to cook this, salt it well, and it is the best, most tender popcorn I have ever eaten. I add a tiny bit of butter to mine, but don\'t need it. My nine year old daughter didn\'t like popcorn until she reluctantly tried this. After a few bites, she consumed half the bowl!<br /><br />I bought mine at a specialty popcorn shop in Long Grove IL, so I didn\'t have to pay shipping costs, but when it\'s gone, I might have to bite the bullet and order it from here.",Spoiled me for other popcorn'
    ori_test_text =  'I bought this with the whirly pop popcorn popper, and I made some as soon as I got them. It is the best popcorn I\'ve had. I thought I would mind the smaller kernel size, but I didn\'t. There is substantially less hull than regular popcorn. The flavor is great. I suggest using peanut oil for popping popcorn. It gives popcorn a great nutty taste - really delicious. You can buy peanut oil at any oriental (Asian) store. It\'s better than olive oil (olive oil and popcorn is too weird for me). But peanut oil is perfect. You won\'t need butter flavor or anything like that again (artificial butter flavorings are full of chemicals). Now, we pop around three times a day and always crave this for a snack instead of junkfood. Try it and you\'ll be popping like us.'
    #清洗数据
    test_text = test_clean_text(ori_test_text)
    #获得字典
    vocab2int, int2vocab = get_vocab_dict()
    #使用字典来对文本进行编码
    test_text = [vocab2int.get(word, vocab2int['<UNK>']) for word in test_text]

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        #加载模型
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        #获得原模型中变量
        text_input = loaded_graph.get_tensor_by_name('text_input:0')
        predict = loaded_graph.get_tensor_by_name('predict:0')
        text_sequence_length = loaded_graph.get_tensor_by_name('text_sequence_length:0')
        summary_sequence_length = loaded_graph.get_tensor_by_name('summary_sequence_length:0')

        feed = {
            text_input:[test_text] * batch_size,
            text_sequence_length:[len(test_text)] * batch_size,
            summary_sequence_length:[len(test_text)]
        }

        #获得预测
        pred_summary = sess.run([predict], feed_dict=feed)[0][0]

        #打印
        print('------------the text is:----------------')
        print(ori_test_text)
        print('------------the summary is:-------------')
        print(" ".join([int2vocab[i] for i in pred_summary if i != vocab2int["<PAD>"] and i != vocab2int["<EOS>"]]))

