"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
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
    #reshape to 4D arrays
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    #create data iterator with NDArrayIter
    (train_lbl, train_img) = read_data('train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    print "val_label_len:%d val_img_data_len:%d train_label_len:%d train_img_data_len:%d"%(len(val_lbl),len(val_img),len(train_lbl),len(train_img))                              
    train = mx.io.NDArrayIter(
        data         = to4d(train_img), 
        label        = train_lbl, 
        batch_size   = args.batch_size, 
        shuffle      = True
    )
    val = mx.io.NDArrayIter(
        data         = to4d(val_img), 
        label        = val_lbl, 
        batch_size   = args.batch_size
    )
    return (train, val)

#a simple multilayer perceptron
def get_symbol(num_classes=10, **kwargs):
    data= mx.symbol.Variable('data')
    # first conv layer
    conv1= mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1= mx.sym.Activation(data=conv1, act_type="tanh")
    pool1= mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    # second conv layer
    conv2= mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2= mx.sym.Activation(data=conv2, act_type="tanh")
    pool2= mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc layer
    flatten= mx.sym.Flatten(data=pool2)
    fc1= mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3= mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2= mx.sym.FullyConnected(data=tanh3, num_hidden=10)
    # softmax loss
    lenet= mx.sym.SoftmaxOutput(data=fc2, name='softmax')   
    shape = {"data":(64, 1, 28, 28)}
    mx.viz.plot_network(symbol=lenet, shape=shape).view()
    return lenet

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))
    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d"%(args.model_prefix, rank))

#----------------------------------------------------------------------
def main():
    #load network
    symbol_net = get_symbol(**vars(args))
    kv = mx.kvstore.create(args.kv_store)
    # data iterators
    (train, val) = get_mnist_iter(args, kv)   
    arg_params, aux_params = (None, None)
    if(args.retrain):
        sym, arg_params, aux_params = _load_model(args, kv.rank)   
    # save model
    checkpoint = _save_model(args, kv.rank)  
    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)
    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = symbol_net
    )
    optimizer_params = {
            'learning_rate': lr,
            'momentum'     : args.mom,
            'wd'           : args.wd,
            'lr_scheduler' : lr_scheduler
    }
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    # evaluation metrices
    eval_metrics = ['accuracy']
    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    # run
    model.fit(
        train_data         = train,
        eval_data          = val,
        eval_metric        = eval_metrics,        
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True)    

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist")
    parser.add_argument('--num-classes', type=int, default=10, help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,help='the number of training examples')
    parser.add_argument('--gpus', type=str, default =None, help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    parser.add_argument('--kv-store', type=str, default='local', help='key-value store type')
    parser.add_argument('--num-epochs', type=int, default=10, help='max num of epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate, e.g. 0.01,0.05,0.1,0.2')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-step-epochs', type=str, default='10', help='the epochs to reduce the lr, e.g. 10,30,60')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer type')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch size, e.g. 16,32,64,128')
    parser.add_argument('--disp-batches', type=int, default=100, help='show progress for every n batches')
    parser.add_argument('--model-prefix', type=str, default='./model/mnist', help='model prefix')
    parser.add_argument('--retrain', type=bool, default=False, help='true means continue training from load-epoch')
    parser.add_argument('--load-epoch', type=int, default=0, help='load the model on an epoch using the model-load-prefix')
    args = parser.parse_args()
    logging.info('arguments %s', args)
    main()
