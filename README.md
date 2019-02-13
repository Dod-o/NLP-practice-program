# NLP练手项目路线

## 简介

各种NLP练手项目，贯彻注释比代码多的风格，学起来更带劲。[博客地址](http://www.pkudodo.com/)          
        
## 版本：
```TensorFlow 1.4.0```

## 包含内容
### 1.word2vec词嵌入
**词嵌入**：基于skip-gram训练词嵌入矩阵，每个词由300维向量表示，相同意义的词向量相似。     
在NLP处理中通常会采用词嵌入来表示每个词。     
[-->项目入口](https://github.com/Dod-o/NLP-practice-program/tree/master/1.skip_gram)          
[-->代码详解_视频入口](https://www.bilibili.com/video/av42442500)    

##### 运行结果 （选取其中一个单词为例，根据词嵌入矩阵计算邻近词）
>**训练前**：    
hemoglobin -->  alden, vive, deviations, dlp, taj, beauvoir, pillow, allying      
有道翻译结果：血红蛋白  --> 奥尔登，vive，偏差，dlp，泰姬陵，波伏娃，枕头，结盟     
**训练后**：     
hemoglobin --> ligand, molecules, ligands, photosynthesis, aerobic, enzyme, pancreatic, chlorophyll     
有道翻译结果：血红蛋白 --> 配体、分子、配体、光合作用、需氧、酶、胰腺、叶绿素

&nbsp; 
### 2.文本生成    
**风格仿写**：学习哈利波特1-7全文，训练结束后给定起始单词（下方运行结果中，给定的起始单词为'Hi, '）,由模型自主生成哈利波特风格的句子。      
[-->项目入口](https://github.com/Dod-o/NLP-practice-program/tree/master/2.harry_potter_lstm)    
[-->代码详解_视频入口](https://www.bilibili.com/video/av42442500/?p=3)    
##### 运行结果
>Hi, he was nearly off at Harry to say the time that and she had been back to his staircase of the too the Hermione?

&nbsp; 
### 3.字母排序     
seq2seq最基础应用，给定单词如bca，使用seq2seq排序为abc。      
[-->项目入口](https://github.com/Dod-o/NLP-practice-program/tree/master/3.seq2seq_alphabetical%20order)    
##### 运行结果
>the input is: hello     
the output is: ['e', 'h', 'l', 'l', 'o']      
    
&nbsp; 
### 4.摘要生成     
seq2seq应用，给定一段话，自动生成摘要。      
[-->项目入口](https://github.com/Dod-o/NLP-practice-program/tree/master/4-seq2seq_summary_burner)    
##### 运行结果
>------------the text is:----------------      
Use olive oil to cook this, salt it well, and it is the best, most tender popcorn I have ever eaten. I add a tiny bit of butter to mine, but don't need it. My nine year old daughter didn't like popcorn until she reluctantly tried this. After a few bites, she consumed half the bowl!     
I bought mine at a specialty popcorn shop in Long Grove IL, so I didn't have to pay shipping costs, but when it's gone, I might have to bite the bullet and order it from here.",Spoiled me for other popcorn      
------------the summary is:-------------     
best tasting popcorn ever     
     
