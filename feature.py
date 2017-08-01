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

#----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-prefix', default='./model/mnist', help='The trained model')
    parser.add_argument('--epoch', type=int, default=5, help='The epoch number of model')
    parser.add_argument('--batch-size', type=int, default=1, help='the batch size')
    args = parser.parse_args()
    logging.info(args)
    
    test_data = get_mnist_iter(args)
    model_load = mx.model.FeedForward.load(args.model_prefix, args.epoch)
    
    internals = model_load.symbol.get_internals()

    print internals.list_outputs()
   
    feature_symbol = internals["softmax_output"] # need to know the feature name
    feature_extractor= mx.model.FeedForward(ctx=mx.cpu(),symbol=feature_symbol,
                                            arg_params=model_load.arg_params,aux_params=model_load.aux_params,allow_extra_params=True)    
    feature = feature_extractor.predict(test_data) 
    print feature.shape
    print feature[0]
if __name__ == '__main__':
    main()