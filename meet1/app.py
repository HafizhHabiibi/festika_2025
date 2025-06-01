import cv2
# # print(cv2.__version__)

# img = cv2.imread("./cat.jpg")

# resize = cv2.resize(img, (1280, 720)) #rresize

# flip = cv2.flip(resize, 0) #flip

# rotate1 = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)
# rotate2 = cv2.rotate(flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
# rotate3 = cv2.rotate(flip, cv2.ROTATE_180)

# (w,h) = resize.shape[:2]
# center = (w//2, h//2)
# rotation = cv2.getRotationMatrix2D(center, 30, 1.0)
# rotation_image = cv2.warpAffine(resize, rotation, (w,h)) #custom rotate

# crop = resize[100:400, 100:400]

# # cv2.imshow("Cat",rotate1)
# # cv2.imshow("Cat",rotate2)
# # cv2.imshow("Cat",rotate3)

# cv2.imshow("Cat", crop)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

# VIDEO
video = cv2.VideoCapture("./vid1.mkv")

while True:
    ret, frame = video.read()

    if not ret:
        break
    cv2.imshow("Test", frame)
    if cv2.waitKey(25) & 0xff == ord('q'):
        break
video.release()
