import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import math

left_clicks = list()


def mouse_callback(event, x, y, flags, params):

    #left-click event value is 1
    if event == 1:
        global left_clicks

        #store the coordinates of the left-click event
        left_clicks.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print (left_clicks)                                     


pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# other_stream, other_format = rs.stream.infrared, rs.format.y8
other_stream, other_format = rs.stream.color, rs.format.rgb8
config.enable_stream(other_stream, 640, 480, other_format, 30)

profile = pipe.start(config)

#skip 5 frames so exposure can auto adjust
for x in range(50):
  pipe.wait_for_frames()

frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

pipe.stop()
print("frames captured")

color = np.asanyarray(color_frame.get_data())
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = [12,6]
#plt.imshow(color)

colorizer = rs.colorizer()
colorizedDepth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
#plt.imshow(colorizedDepth)

align = rs.align(rs.stream.color)
frameset = align.process(frameset)

aligned_depth_frame = frameset.get_depth_frame()
 
colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

images = np.stack((color, colorized_depth))

f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(color)
f.add_subplot(1,2, 2)
plt.imshow(colorized_depth)
plt.show(block=True)
#plt.imshow(color)
#plt.imshow(colorizedDepth)


#while True:
key = cv2.waitKey(1)

cv2.imwrite('tempImage1.png', cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR)) #Use this to get a selection of points on colorized depth image

#cv2.imwrite('tempImage1.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))            #Use this to get a selection of points on color image

print('select points and press Enter')


img = cv2.imread("tempImage1.png")
scale_width = 640 / img.shape[1]
scale_height = 480 / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)


cv2.setMouseCallback('image', mouse_callback)

cv2.imshow('image', img)
cv2.waitKey(0)           
if key == (27, ord("q")):
    cv2.destroyAllWindows('image')
#measuring distance
#input("press enter to continue")
#cv2.line(img,(0,0),(200,300),(255,255,255),50)
y1 = left_clicks[0][0]
x1 = left_clicks[0][1]
y2 = left_clicks[1][0]
x2 = left_clicks[1][1]

#cv2.line(img,(y1,x1),(y2,x2),(0,0,255),2)

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cropped_img = gray_img[x1:x2,y1:y2]
edges = cv2.Canny(cropped_img,100,200)
plt.imshow(edges)
