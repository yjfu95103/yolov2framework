import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'voc.names'
    elif m.num_classes == 80:
        namesfile = 'coco.names'
    else:
        namesfile = 'names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    img.save("../example/data/input.jpg")
    sized = img.resize((m.width, m.height))
    #print(sized)
    #imageoutput(sized,savename='convolution.jpg')

    for i in range(1):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, '../example/data/predictions.jpg', class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'voc.names'
    elif m.num_classes == 80:
        namesfile = 'coco.names'
    else:
        namesfile = 'names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions1.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions2.jpg', class_names=class_names)

def rgb_image(imgfile):
    import cv2
    import numpy as np

    #read image
    src = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
    print(src.shape)
    # extract red channel
    red_channel = src[:,:,2]
    red_img = np.zeros(src.shape)
    red_img[:,:,2] = red_channel
    cv2.imwrite('../example/data/input/r.png',red_img) 

    green_channel = src[:,:,1]
    green_img = np.zeros(src.shape)
    green_img[:,:,1] = green_channel
    cv2.imwrite('../example/data/input/g.png',green_img) 

    blue_channel = src[:,:,0]
    blue_img = np.zeros(src.shape)
    blue_img[:,:,0] = blue_channel
    cv2.imwrite('../example/data/input/b.png',blue_img) 

# def dete3 


if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        rgb_image(imgfile)

        # net = load_net((sys.argv[2]).encode('utf-8'), (sys.argv[3]).encode('utf-8'), 0)
        # meta = 'detect'#load_meta((sys.argv[1]).encode('utf-8'))
        # r = detect(net, meta, (sys.argv[4]).encode('utf-8'))
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
