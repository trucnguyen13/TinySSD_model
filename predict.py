from lib import *
from model import TinySSD

modelfile = "model/net.params"

if not os.path.exists('img'):
    os.makedirs('img')

imgfile = "img/pikachu.jpg"
imgURL = "https://raw.githubusercontent.com/d2l-ai/d2l-en/master/img/pikachu.jpg"
urllib.request.urlretrieve(imgURL, imgfile)

img = image.imread(imgfile)
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)


device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.load_parameters(modelfile, ctx=device)


def predict(X , net):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = npx.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X, net)


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    d2l.plt.show()

display(img, output, threshold=0.3)