from model import TinySSD
from loss import *
from eval import *
from dataloader import *


batch_size = 32
train_iter, _ = load_data_pikachu(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2, 'wd': 5e-4})


num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])

if not os.path.exists('model'):
    os.makedirs('model')
modelfile = "model/net.params"

def train_model(net, train_iter, criterion, optimizer, num_epochs, animator, cls_eval,  bbox_eval, device, modelfile):                        
    for epoch in range(num_epochs):
        # accuracy_sum, mae_sum, num_examples, num_labels
        metric = d2l.Accumulator(4)
        train_iter.reset()  # Read data from the start.
        for batch in train_iter:
            print("epoch: ", epoch + 1)
            timer.start()
            X = batch.data[0].as_in_ctx(device)
            Y = batch.label[0].as_in_ctx(device)
            with autograd.record():
                # Generate multiscale anchor boxes and predict the category and
                # offset of each
                anchors, cls_preds, bbox_preds = net(X)
                # Label the category and offset of each anchor box
                bbox_labels, bbox_masks, cls_labels = npx.multibox_target( \
                    anchors, Y, cls_preds.transpose(0, 2, 1))
                # Calculate the loss function using the predicted and labeled
                # category and offset values
                l = criterion(cls_preds, cls_labels, bbox_preds, bbox_labels, \
                                bbox_masks)
            l.backward()
            optimizer.step(batch_size)
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size, \
                        bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                        bbox_labels.size)
        cls_err, bbox_mae = 1-metric[0]/metric[1], metric[2]/metric[3] 
        animator.add(epoch+1, (cls_err, bbox_mae))
        net.save_parameters(modelfile)

    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{train_iter.num_image/timer.stop():.1f} examples/sec on ' \
            f'{str(device)}')
            


train_model(net = net, train_iter = train_iter, criterion = calc_loss, optimizer = trainer, num_epochs = num_epochs, \
 animator = animator, cls_eval = cls_eval,  bbox_eval = bbox_eval, device = device, modelfile = modelfile)












