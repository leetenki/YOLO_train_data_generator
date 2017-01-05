import cv2
import os
import glob
import numpy as np
from PIL import Image

# src_imageの背景画像に対して、overlay_imageのalpha画像を貼り付ける。pos_xとpos_yは貼り付け時の左上の座標
def overlay(src_image, overlay_image, pos_x, pos_y):
    # オーバレイ画像のサイズを取得
    ol_height, ol_width = overlay_image.shape[:2]

    # OpenCVの画像データをPILに変換
    # BGRAからRGBAへ変換
    src_image_RGBA = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    overlay_image_RGBA = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)

    #　PILに変換
    src_image_PIL=Image.fromarray(src_image_RGBA)
    overlay_image_PIL=Image.fromarray(overlay_image_RGBA)

    # 合成のため、RGBAモードに変更
    src_image_PIL = src_image_PIL.convert('RGBA')
    overlay_image_PIL = overlay_image_PIL.convert('RGBA')

    # 同じ大きさの透過キャンパスを用意
    tmp = Image.new('RGBA', src_image_PIL.size, (255, 255,255, 0))
    # 用意したキャンパスに上書き
    tmp.paste(overlay_image_PIL, (pos_x, pos_y), overlay_image_PIL)
    # オリジナルとキャンパスを合成して保存
    result = Image.alpha_composite(src_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)

# 画像周辺のパディングを削除
def delete_pad(image): 
    orig_h, orig_w = image.shape[:2]
    mask = np.argwhere(image[:, :, 3] > 128) # alphaチャンネルの条件、!= 0 や == 255に調整できる
    (min_y, min_x) = (max(min(mask[:, 0])-1, 0), max(min(mask[:, 1])-1, 0))
    (max_y, max_x) = (min(max(mask[:, 0])+1, orig_h), min(max(mask[:, 1])+1, orig_w))
    return image[min_y:max_y, min_x:max_x]

# 画像を指定した角度だけ回転させる
def rotate_image(image, angle):
    orig_h, orig_w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((orig_h/2, orig_w/2), angle, 1)
    return cv2.warpAffine(image, matrix, (orig_h, orig_w))

# 画像をスケーリングする
def scale_image(image, scale):
    orig_h, orig_w = image.shape[:2]
    return cv2.resize(image, (int(orig_w*scale), int(orig_h*scale)))

# 背景画像から、指定したhとwの大きさの領域をランダムで切り抜く
def random_sampling(image, h, w): 
    orig_h, orig_w = image.shape[:2]
    y = np.random.randint(orig_h-h+1)
    x = np.random.randint(orig_w-w+1)
    return image[y:y+h, x:x+w]

# 画像をランダムに回転、スケールしてから返す
def random_rotate_scale_image(image):
    image = rotate_image(image, np.random.randint(360))
    image = scale_image(image, 1 + np.random.rand() * 2) # 1 ~ 3倍
    return delete_pad(image)

# overlay_imageを、src_imageのランダムな場所に合成して、そこのground_truthを返す。
def random_overlay_image(src_image, overlay_image):
    src_h, src_w = src_image.shape[:2]
    overlay_h, overlay_w = overlay_image.shape[:2]
    y = np.random.randint(src_h-overlay_h+1)
    x = np.random.randint(src_w-overlay_w+1)
    bbox = ((x, y), (x+overlay_w, y+overlay_h))
    return overlay(src_image, overlay_image, x, y), bbox

# 4点座標のbboxをyoloフォーマットに変換
def yolo_format_bbox(image, bbox):
    orig_h, orig_w = image.shape[:2]
    center_x = (bbox[1][0] + bbox[0][0]) / 2 / orig_w
    center_y = (bbox[1][1] + bbox[0][1]) / 2 / orig_h
    w = (bbox[1][0] - bbox[0][0]) / orig_w
    h = (bbox[1][1] - bbox[0][1]) / orig_h
    return(center_x, center_y, w, h)

base_path = os.getcwd()
fruit_files = glob.glob("orig_images/*")
fruits = []
labels = []
for fruit_file in fruit_files:
    labels.append(fruit_file.split("/")[-1].split(".")[0])
    fruits.append(cv2.imread(fruit_file, cv2.IMREAD_UNCHANGED))
background_image = cv2.imread("background.jpg")

# write label file
with open("label.txt", "w") as f:
    for label in labels:
        f.write("%s\n" % (label))

background_height, background_width = (416, 416)
train_images = 10000
test_images = 2000

# train用の画像生成
for i in range(train_images):
    sampled_background = random_sampling(background_image, background_height, background_width)

    class_id = np.random.randint(len(labels))
    fruit = fruits[class_id]
    fruit = random_rotate_scale_image(fruit)

    result, bbox = random_overlay_image(sampled_background, fruit)
    yolo_bbox = yolo_format_bbox(result, bbox)

    # 画像ファイルを保存
    image_path = "%s/images/train_%s_%s.jpg" % (base_path, i, labels[class_id])
    cv2.imwrite(image_path, result)

    # 画像ファイルのパスを追記
    with open("train.txt", "a") as f:
        f.write("%s\n" % (image_path))

    # ラベルファイルを保存
    label_path = "%s/labels/train_%s_%s.txt" % (base_path, i, labels[class_id]) 
    with open(label_path, "w") as f:
        f.write("%s %s %s %s %s" % (class_id, yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3]))

    print("train image", i, labels[class_id], yolo_bbox)

# test用の画像生成
for i in range(test_images):
    sampled_background = random_sampling(background_image, background_height, background_width)

    class_id = np.random.randint(len(labels))
    fruit = fruits[class_id]
    fruit = random_rotate_scale_image(fruit)

    result, bbox = random_overlay_image(sampled_background, fruit)
    yolo_bbox = yolo_format_bbox(result, bbox)

    # 画像ファイルを保存
    image_path = "%s/images/test_%s_%s.jpg" % (base_path, i, labels[class_id])
    cv2.imwrite(image_path, result)

    # 画像ファイルのパスを追記
    with open("test.txt", "a") as f:
        f.write("%s\n" % (image_path))

    # ラベルファイルを保存
    label_path = "%s/labels/test_%s_%s.txt" % (base_path, i, labels[class_id]) 
    with open(label_path, "w") as f:
        f.write("%s %s %s %s %s" % (class_id, yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3]))

    print("test image", i, labels[class_id], yolo_bbox)
