import sys
import csv
import os
sys.path.append("/home/oleg/Projects/caffe/python/")

import caffe
import numpy as np
from tqdm import tqdm

test_images = list()
with open("/home/oleg/SOS/imgIdx.csv", "r") as file:
    csvreader = csv.reader(file)
    skip = True
    for line in csvreader:
        if skip:
            skip = False
            continue

        if line[2] == '0':
            continue
        path = os.path.join('/home/oleg/SOS/img', line[0])
        test_images.append((path, int(line[1])))

results = {'Stohastic': {'GoogleNet': {}, 'AlexNet': {}, 'VGG16': {}}, 'Zeroing': {'GoogleNet': {}, 'AlexNet': {}, 'VGG16': {}}}

for stohastic in [True, False]:
    for frac_bits in range(23, 7, -1):
        for name in ("GoogleNet", "AlexNet", "VGG16"):
            net_proto = os.path.join('/home/oleg/SOS/', name, 'description.prototxt')
            model = os.path.join('/home/oleg/SOS', name, 'model.caffemodel')
            net = caffe.Net(net_proto, model, caffe.TEST)
            caffe.set_clip_params(0, frac_bits, stohastic)

            transformer = caffe.io.Transformer({'data': net.blobs['data'].shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_channel_swap('data', (2, 1, 0))
            transformer.set_raw_scale('data', 255)
            transformer.set_mean('data', np.array([103.939, 116.779, 123.68], dtype='f4'))

            positive_answers = 0
            counter = 0

            description_str = "{}/{}/{}".format(name, frac_bits, "Stohastic" if stohastic else "Zeroing")

            for img in tqdm(test_images, desc=description_str):
                image = caffe.io.load_image(img[0])
                net.blobs['data'].data[...] = transformer.preprocess('data', image)
                out = net.forward()
                label = int(img[1])
                if out['prob'].argmax() == label:
                    positive_answers = positive_answers + 1
                counter += 1

            accuracy = float(positive_answers) / len(test_images)
            print '{} has {} positive answers ({})'.format(description_str, positive_answers, accuracy)
            if stohastic:
                results['Stohastic'][name][frac_bits] = accuracy
            else:
                results['Zeroing'][name][frac_bits] = accuracy

np.save('./results.npy', results)
print results
