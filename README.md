# Pyaq

PyaqはPythonのみで実装された囲碁プログラムです。  
このプログラムは深層学習のチュートリアルとして、囲碁のニューラルネットワークモデルを学習させ、実際に対局することを目的としています。  

![top](https://user-images.githubusercontent.com/32036527/36086412-90005ab6-100f-11e8-912b-fdf30c61b2ef.png)  

具体的には次の内容を行います。  
- [TensorFlow](https://www.tensorflow.org/)で９路盤の棋譜を学習する
- 学習したモデルを使って対局する

囲碁の対局や深層学習のための必要最小限の実装となっており、学習・実行のすべてのコードを合わせて1000行程度です。より発展的に学習したい方はソースコードを読んでみると良いでしょう。もちろんプルリクエストも歓迎です。  

## １． 準備する

下記の環境を例に説明を進めます。  
- Ubuntu 16.04
- Python 2.7
- TensorFlow  
  
TensorFlowの導入は[UbuntuにTensorFlowをインストール](https://qiita.com/yudsuzuk/items/092c38fee18e4484ece9)を参考にしてください。  
TensorFlowでGPUを用いる場合は  
- [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive)
- [cuDNN v7.0](https://developer.nvidia.com/cudnn)
  
をインストールしておく必要があります。また、nVidia製の[CUDA Capability](https://developer.nvidia.com/cuda-gpus)3.5以上のグラフィックボードが必要です。  
CUDA導入は[CUDA 8.0とcuDNN 6をUbuntu 16.04LTSにインストールする](https://qiita.com/JeJeNeNo/items/05e148a325192004e2cd)などを参考にしてください（注：リンク先の場合とバージョンが異なります）。  

次に、ソースコードをダウンロードします。  
```
$ git clone https://github.com/ymgaq/Pyaq
```
または右上の「Clone or download」から手動でダウンロードすることもできます。  
これで準備は完了です。  

すぐ遊びたい人は、学習済みのデータファイル```Pyaq/pre_train/model.ckpt```を```Pyaq/```にコピーし、「4.GoGuiで対局する」に進んでください。  

では試しにテスト対戦が動作するかを確認してみましょう。  

```
$ ./pyaq.py --self --random
```

次のような出力が出れば成功です。「2.学習する」に進みましょう。  

```
   A  B  C  D  E  F  G  H  J 
 9 .  X  X  X  X  X  X  .  X  9
 8 X  X  X  .  X  X  X  X  X  8
 7 O  O  X  X  X  O  O  X  O  7
 6 O  O  O  O  O  .  O  O  O  6
 5 O  O  X  O  O  O  O  .  O  5
 4 X  X  X  O  .  O  .  O  O  4
 3 .  X  O  O  O  O  O  O  .  3
 2 X  X  X  X  X  O  O  O  O  2
 1 X  .  X  O  O  O  O  O  .  1
   A  B  C  D  E  F  G  H  J 

   A  B  C  D  E  F  G  H  J 
 9 .  X  X  X  X  X  X  .  X  9
 8 X  X  X  .  X  X  X  X  X  8
 7 O  O  X  X  X  O  O  X  O  7
 6 O  O  O  O  O  .  O  O  O  6
 5 O  O  X  O  O  O  O  .  O  5
 4 X  X  X  O  .  O  .  O  O  4
 3 .  X  O  O  O  O  O  O  .  3
 2 X  X  X  X  X  O  O  O  O  2
 1 X  .  X  O  O  O  O  O  .  1
   A  B  C  D  E  F  G  H  J 

result: W+16.0
```

## 2. 学習する

まず、学習用ファイルを展開します。  

```
$ cd Pyaq
$ unzip sgf.zip
```

9路盤の棋譜ファイル（*.sgf）を用いて学習を行います。次のコマンドを実行すると、学習が始まります。  

```
$ ./pyaq.py --learn
```

GPUなしで学習させたい場合は```--cpu```オプションを追加してください。  
（ただし、CPUのみの学習は十分にテストされていません。）  

```
$ ./pyaq.py --learn --cpu
```

次のように学習ログが展開されます。また、log.txtにも同じ内容が記録されます。  
GPUの性能にもよりますが、大体3〜4時間で学習が完了します。 CPUのみの場合は3日程度かかります。  

```
imported 34572 sgf files.
converting ...
learning rate=0.0003
progress: 0.10[%] 14.3[sec]
progress: 0.20[%] 13.3[sec]
progress: 0.30[%] 13.3[sec]
progress: 0.40[%] 13.4[sec]
progress: 0.50[%] 13.3[sec]
progress: 0.60[%] 13.4[sec]
progress: 0.70[%] 13.3[sec]
progress: 0.80[%] 13.3[sec]
progress: 0.90[%] 13.2[sec]
progress: 1.00[%] 13.2[sec]
progress: 1.10[%] 13.3[sec]
progress: 1.20[%] 13.3[sec]
progress: 1.30[%] 13.3[sec]
progress: 1.40[%] 13.2[sec]
progress: 1.50[%] 13.2[sec]
progress: 1.60[%] 13.2[sec]
progress: 1.70[%] 13.3[sec]
progress: 1.80[%] 13.3[sec]
progress: 1.90[%] 13.2[sec]
progress: 2.00[%] 13.2[sec]
progress: 2.10[%] 13.2[sec]
progress: 2.20[%] 13.4[sec]
progress: 2.30[%] 13.4[sec]
progress: 2.40[%] 13.2[sec]
progress: 2.50[%] 13.3[sec]
train: policy=46.95[%]  value=0.469
test : policy=47.13[%]  value=0.469

progress: 2.60[%] 15.5[sec]
progress: 2.70[%] 13.4[sec]
```

2.5%ごとにtestデータの評価を行います。 ```policy```は棋譜の次の手とニューラルネットワークが出力する手との一致率、```value```は棋譜の勝敗とネットワークが出力する評価値（-1~+1）の誤差（Mean Squared Error）を表します。 最終的に、testデータでpolicyが57%、valueが0.36程度になるようです。  
学習が完了すると、パラメータファイル```model.ckpt```が保存されます。  

ネットワークモデルの```BLOCK_CNT```や```FILTER_CNT```、または盤面の```KEEP_PREV_CNT```などを変更したり、モデルの形を変えたり、オリジナルの棋譜データを使用することで、より強力なパラメータを生成できる可能性があります。 興味がある方は、あなただけの最強のネットワーク作りに挑戦してみましょう。  

## 3. 自己対戦をさせてみる（コンソール）

コンソール上で学習したモデルを使って、まず探索なしの自己対戦をさせてみます。  

```
$ ./pyaq.py --self --quick --cpu
```

探索なしの場合の対戦結果が得られます。  

```
   A  B  C  D  E  F  G  H  J 
 9 .  .  .  .  .  O  O  X  .  9
 8 .  .  O  O  .  O  X  X  X  8
 7 .  O  X  X  O  O  O  X  .  7
 6 .  .  .  O  X  O  X  X  .  6
 5 O  O  .  O  X  X  O  .  .  5
 4[X] X  O  O  O  X  O  .  .  4
 3 X  X  X  O  X  X  X  .  .  3
 2 X  .  X  O  X  .  .  .  .  2
 1 .  X  O  O  O  X  .  .  .  1
   A  B  C  D  E  F  G  H  J 

   A  B  C  D  E  F  G  H  J 
 9 .  .  .  .  .  O  O  X  .  9
 8 .  .  O  O  .  O  X  X  X  8
 7 .  O  X  X  O  O  O  X  .  7
 6 .  .  .  O  X  O  X  X  .  6
 5 O  O  .  O  X  X  O  .  .  5
 4 X  X  O  O  O  X  O  .  .  4
 3 X  X  X  O  X  X  X  .  .  3
 2 X  .  X  O  X  .  .  .  .  2
 1 .  X  O  O  O  X  .  .  .  1
   A  B  C  D  E  F  G  H  J 

   A  B  C  D  E  F  G  H  J 
 9 .  .  .  .  .  O  O  X  .  9
 8 .  .  O  O  .  O  X  X  X  8
 7 .  O  X  X  O  O  O  X  .  7
 6 .  .  .  O  X  O  X  X  .  6
 5 O  O  .  O  X  X  O  .  .  5
 4 X  X  O  O  O  X  O  .  .  4
 3 X  X  X  O  X  X  X  .  .  3
 2 X  .  X  O  X  .  .  .  .  2
 1 .  X  O  O  O  X  .  .  .  1
   A  B  C  D  E  F  G  H  J 


result: Draw
```

次に、探索ありの自己対戦をしてみましょう。  

```
$ ./pyaq.py --self --byoyomi=3
```

GPUなしの場合は```--cpu```オプションを追加してください。  

```
$ ./pyaq.py --self --byoyomi=3 --cpu
```

１手３秒で対局が進行します。  

```
move count=3: left time=0.0[sec] evaluated=104
|move|count  |rate |value|prob | best sequence
|D5  |   1114| 54.7| 56.3| 90.4| D5 ->C5 ->C4 ->E5 ->D6 ->E6 ->E7 ->E4 
|E4  |    150| 51.3| 55.2|  0.8| E4 ->E3 ->D5 ->C5 ->E5 ->F3 ->C4 ->D3 
|F4  |     20| 51.2| 54.3|  0.8| F4 ->D6 ->D7 
|D6  |      1| 48.1| 48.1|  3.0| D6 
|C6  |      1| 46.0| 46.0|  2.3| C6 
|C3  |      1| 44.4| 44.4|  1.8| C3 
   A  B  C  D  E  F  G  H  J 
 9 .  .  .  .  .  .  .  .  .  9
 8 .  .  .  .  .  .  .  .  .  8
 7 .  .  .  .  .  .  .  .  .  7
 6 .  .  .  .  .  X  .  .  .  6
 5 .  .  . [X] .  .  .  .  .  5
 4 .  .  .  O  .  .  .  .  .  4
 3 .  .  .  .  .  .  .  .  .  3
 2 .  .  .  .  .  .  .  .  .  2
 1 .  .  .  .  .  .  .  .  .  1
   A  B  C  D  E  F  G  H  J 
```

思考ログの内容は次の通りです。  
- ```move count``` 手数
- ```left time``` 残り時間
- ```evaluated``` この思考で評価された盤面の数
- ```move``` 候補手
- ```count``` 探索回数
- ```rate``` 手番側からみた勝率
- ```value``` 候補手を着手した場合の評価値
- ```prob``` 候補手の確率
- ```best sequence``` 候補手の後の読み筋

また、pyaq.pyのコマンドラインオプションは以下の通りです。  
- ```--cpu``` CPUのみを使用する
- ```--learn``` 棋譜から学習する
- ```--self``` コンソールで自己対戦を行う
- ```--random``` ランダムに着手する
- ```--quick``` 確率最大の手を選択する（探索しない）
- ```--clean``` 最後まで打ち切る（探索ありの場合のみ）
- ```--main_time=600``` 持ち時間 10分を設定
- ```--byoyomi=10``` 秒読み 10秒を設定

## 4. GoGuiで対局する

学習をしていない人は、学習済みのデータファイル```Pyaq/pre_train```にある```model.ckpt```を```Pyaq/```にコピーしてください。  

[GoGui](https://sourceforge.net/projects/gogui/files/gogui/1.4.9/)を使ってGUIでの対局を行います。  
メニュー＞対局＞碁盤サイズを「9」に設定した後、
メニュー＞プログラム＞新規プログラムから「コマンド」と「ワーキングディレクトリ」を登録します。  

![resister](https://user-images.githubusercontent.com/32036527/36086431-acdf1168-100f-11e8-9127-adc138b3fa3d.png)  

起動したらGUIで対局することができます。 思考ログはメニュー＞ツール＞GTPシェルから見ることができます。 

![top](https://user-images.githubusercontent.com/32036527/36086412-90005ab6-100f-11e8-912b-fdf30c61b2ef.png)   

## ライセンス
[MITライセンス](https://github.com/ymgaq/Pyaq/blob/master/LICENSE)  
Author: [Yu Yamaguchi](https://twitter.com/ymg_aq)  
