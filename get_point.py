import cv2

list_of_points = []


# Get the mouse position
def get_point(event, x, y, flags, param):
    global list_of_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        list_of_points.append((x, y))
        cv2.imshow('image', image)


# Create a window
scale = 0.5
# cap = cv2.VideoCapture("rtsp://admin:Namcao123$@192.168.1.69/Streaming/Channels/1")
# cap = cv2.VideoCapture(r'/home/www/Downloads/2.png')
# ret, image = cap.read()
image = cv2.imread('/home/huyquang/Downloads/2.png')
image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

cv2.namedWindow('image')
cv2.setMouseCallback('image', get_point)
cv2.imshow('image', image)
key = cv2.waitKey()
if key == ord('q'):
    print(list_of_points)
    cv2.destroyAllWindows()
elif key == ord('s'):
    with open(r'head.txt', 'w+') as f:
        for point in list_of_points:
            f.write(str(int(point[0] / scale)) + ' ' + str(int(point[1] / scale)) + '\n')
    cv2.destroyAllWindows()
