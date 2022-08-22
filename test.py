import cv2

path = "/home/huyquang/Downloads/2.png"
image = cv2.imread(path)
print(image.shape)

points = []
with open("head.txt", "r") as f:
    for line in f.readlines():
        if not line.strip():
            continue
        x, y = list(map(int, line.split(" ")))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

cv2.imshow('Image', image)
key = cv2.waitKey(0)
