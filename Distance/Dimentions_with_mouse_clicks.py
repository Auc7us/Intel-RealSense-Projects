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
for x in range(5):
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

cv2.imwrite('tempImage1.png', cv2.cvtColor(colorized_depth, cv2.COLOR_RGB2BGR))

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

cv2.line(img,(y1,x1),(y2,x2),(0,0,255),2)

depth = np.asanyarray(aligned_depth_frame.get_data()) 
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

depth_1 = depth[x1,y1].astype(float)
depth_2 = depth[x2,y2].astype(float)

depthZ1 = depth_1*depth_scale
depthZ2 = depth_2*depth_scale

print("depth of point 1 is ",depthZ1)
print('depth of point 2 is ',depthZ2)

#xScale = depth_scale*((depthZ2-depthZ1)*0.8268367588)
#yScale = depth_scale*((depthZ2-depthZ1)*0.8268367588)


x2=0.8286085518*(((depthZ2-3*depthZ1)*x1)+((depthZ2+depthZ1)*x2))*0.0010000000474974513

y2=0.8286085518*(((depthZ2-3*depthZ1)*y1)+((depthZ2+depthZ1)*y2))*0.0010000000474974513

print("X222222222  ",x2)
print("Y222222222  ",y2)

x1 = 0
y1 = 0
z1 = 0
z2 = depthZ2-depthZ1

print("\n\ndepthScale is ",depth_scale)

coord1 = (x1,y1,z1)
coord2 = (x2,y2,z2)

#textX = (x1+x2)/4
#textY = ((y1+y2)/4)-25

lengthMeasured = math.sqrt(sum([(a-b)**2 for a,b in zip(coord1,coord2)]))

#cv2.putText(img, lengthMeasured, 
#            (int(textX), int(textY)),
#            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

cv2.namedWindow('imageFinal', cv2.WINDOW_NORMAL)
cv2.resizeWindow('imageFinal', window_width, window_height)
cv2.imshow('imageFinal',img)
print("\n\n The Length measured is",lengthMeasured,"\n\n")
cv2.waitKey(0)
if key == (27, ord("q")):
    cv2.destroyAllWindows('imageFinal')
