# ground truth付き画像の合成スクリプト

## ファイル構成
- setup.sh
　ディレクトリ構成を初期化するスクリプト

- background.jpg
　合成元の背景画像。

- orig_images/
　合成対象の各カテゴリの画像ここに保存する。ファイル名はそれぞれラベル名とする(appleならapple.png)。合成用画像なので、全てalphaチャンネル付きのpngファイルである必要がある。
- images/*.jpg
　合成結果として生成された画像がここに保存される。

- labels/*.txt
　合成結果として生成された画像のアノテーション(class_idとground_truth情報)ファイル。このアノテーションファイルは既にyolo専用フォーマットに調整済み。

- train.txt
　画像ファイルへのパス一覧がここに書かれる。

- label.txt
　ラベル名の一覧がここに書かれる。

## ディレクトリ初期化
以下のコマンドでディレクトリ構成を初期化する。

```
./setup.sh
```
ここでは、`images/` `labels/` ディレクトリを空にし、`train.txt`及び`label.txt`ファイルを削除している。

## 合成画像の生成
以下のコマンドで、`orig_images/*.png`と`./background.jpg`を読み込み、そこから指定した枚数分の合成画像を生成する。生成枚数や合成のオプションは`generate_sample.py`のソース内で変更可能。デフォルトでは10000枚生成。background.jpgから416x416領域をランダムで切り取る。合成画像は1〜3倍のランダムスケール、ランダム回転を加えて合成している。

```
python generate_sample.py
```


## 合成画像のチェック
以下のコマンドで、`images/`ディレクトリ内の最初の画像と、対応する`labels/`内のyoloフォーマットのアノテーションを読み込んで描画する。描画した結果ground truth boxが正しく囲まれていれば問題ない。

```
python read_image_by_darknet_format.py
```