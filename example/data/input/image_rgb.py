# import matplotlib.pyplot as plt
# from PIL import Image
# import sys
# import numpy as np

# print (sys.argv)

# img = Image.open('input.jpg')
# data = img.getdata()

# # Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
# r = [(d[0], 0, 0) for d in data]
# g = [(0, d[1], 0) for d in data]
# b = [(0, 0, d[2]) for d in data]

# img.putdata(r)
# plt.figure("channel red")
# plt.imshow(img)
# plt.savefig('dog_red.png')
# # plt.show()
# # img_r = np.array(img)
# # print(img_r)

# img.putdata(g)
# plt.figure("channel green")
# plt.imshow(img)
# plt.savefig('dog_green.png')
# # plt.show()
# # img_g = np.array(img)
# # print(img_g)

# img.putdata(b)
# plt.figure("channel blue")
# plt.imshow(img)
# plt.savefig('dog_blue.png')
# # plt.show()
# # img_b = np.array(img)
# # print(img_b)

import cv2
import numpy as np

#read image
src = cv2.imread('input.jpg', cv2.IMREAD_UNCHANGED)
print(src.shape)

# extract red channel
red_channel = src[:,:,2]
red_img = np.zeros(src.shape)
red_img[:,:,2] = red_channel
cv2.imwrite('r.png',red_img) 

green_channel = src[:,:,1]
green_img = np.zeros(src.shape)
green_img[:,:,1] = green_channel
cv2.imwrite('g.png',green_img) 

blue_channel = src[:,:,0]
blue_img = np.zeros(src.shape)
blue_img[:,:,0] = blue_channel
cv2.imwrite('b.png',blue_img) 