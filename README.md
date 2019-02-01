# NLP练手项目路线

## 简介

各种NLP练手项目，贯彻注释比代码多的风格，学起来更带劲。    

项目文件中包含ckpt文件，可直接运行得到结果，不同项目操作细节请看项目文件内的readme。

[博客地址](http://www.pkudodo.com/)


## 版本：
```TensorFlow 1.4.0```

## 包含内容
### 1.word2vec词嵌入
[基于skip-gram训练词嵌入矩阵](https://github.com/Dod-o/NLP-practice-program/tree/master/1.skip_gram)    
##### 运行结果 （选取其中一个单词为例，根据词嵌入矩阵计算邻近词）
>训练前：    
hemoglobin -->  alden, vive, deviations, dlp, taj, beauvoir, pillow, allying      
有道翻译结果：血红蛋白  --> 奥尔登，vive，偏差，dlp，泰姬陵，波伏娃，枕头，结盟     
      
>训练后：     
hemoglobin --> ligand, molecules, ligands, photosynthesis, aerobic, enzyme, pancreatic, chlorophyll     
有道翻译结果：血红蛋白 --> 配体、分子、配体、光合作用、需氧、酶、胰腺、叶绿素

&nbsp; 
### 2.文本生成
[基于LSTM生成哈利波特风格文本](https://github.com/Dod-o/NLP-practice-program/tree/master/2.harry_potter_lstm)    
##### 运行结果
>Hi, he was nearly off at Harry to say the time that and she had been back to his staircase of the too the Hermione?
     
     
