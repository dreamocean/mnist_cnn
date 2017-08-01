#!/usr/bin/env python
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import numpy as np
import gzip, struct


def read_data(label, image):
    with gzip.open(os.path.join('data',label)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(os.path.join('data',image), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args):
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    print "val_label_len:" + str(len(val_lbl)),
    print "val_img_data_len:" + str(len(val_img)),                         
    val = mx.io.NDArrayIter(
        data         = to4d(val_img), 
        label        = val_lbl, 
        batch_size   = args.batch_size
    )
    return val

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-prefix', default='./model/mnist', help='The trained model')
    parser.add_argument('--epoch', type=int, default=5, help='The epoch number of model')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch size')
    args = parser.parse_args()
    logging.info(args)
    
    test_data = get_mnist_iter(args)
    model_load = mx.model.FeedForward.load(args.model_prefix, args.epoch)
    outputs, data, label = model_load.predict(test_data, return_data = True)
    correct_count = 0.0
    error_count = 0.0    
    print "outputs.shape: " + str(outputs.shape)
    print ('*'*30)
    for i in range(0, outputs.shape[0]):
        predict_label = np.argmax(outputs[i])         
        if label[i] == predict_label:
            iscorrect = True
            correct_count = correct_count + 1.0
        else:
            iscorrect = False
            error_count = error_count + 1.0        
        if i < 100:
            print "max_output: %f  predict_label: %d  ori_abel: %d result: %s"%(np.max(outputs[i]), predict_label, label[i], iscorrect)       
    acc = correct_count/(correct_count + error_count)    
    print "predict accuracy: " + str(acc)
    print 'extract finish'

if __name__ == '__main__':
    main()
