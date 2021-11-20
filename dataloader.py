from lib import *

def load_data_pikachu(batch_size, edge_size=256):
    """Load the pikachu dataset."""
    data_dir = data_dir = 'data/pikachu'
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # The shape of the output image
        shuffle=True,  # Read the dataset in random order
        rand_crop=1,  # The probability of random cropping is 1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter


if __name__ == "__main__":
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_pikachu(batch_size, edge_size)
    batch = train_iter.next()
    print(batch.data[0].shape, batch.label[0].shape)