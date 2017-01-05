import cv2
import glob

file_names = glob.glob("./images/*")
im_id = file_names[0].split("/")[-1].split(".")[0]
im = cv2.imread("./images/%s.jpg" % (im_id))
(im_h, im_w) = im.shape[:2]

in_file = open("./labels/%s.txt" % (im_id))
(label, x, y, w, h) = in_file.readline().split()
x = float(x) * im_w
y = float(y) * im_h
w = float(w) * im_w
h = float(h) * im_h

half_w = w / 2
half_h = h / 2
cv2.rectangle(im, (int(x-half_w), int(y-half_h)), (int(x+half_w), int(y+half_h)), (0, 0, 255), 1)

cv2.imshow("window", im)
cv2.waitKey()
