from scipy import io
import os
import pandas as pd

class DataProcessor:
    def __init__(self, images_root, labels_file, splits_file):
        images = sorted(os.listdir(images_root))
        self.images = [os.path.join(images_root, image) for image in images]
        self.labels = io.loadmat(labels_file)['labels'].flatten()
        self.splits_file = splits_file

    def _splits(self):
        splits = io.loadmat(self.splits_file)

        trnid = splits['trnid'].flatten()
        valid = splits['valid'].flatten()
        tstid = splits['tstid'].flatten()

        return trnid, valid, tstid

    def get_data(self):
        data = pd.DataFrame({
            'image': self.images,
            'label': self.labels,
        })
        trnid, valid, tstid = self._splits()
        train_data = data.iloc[trnid-1]
        valid_data = data.iloc[valid-1]
        test_data = data.iloc[tstid-1]
        return train_data, valid_data, test_data

# dp = DataProcessor(images_root='../data/images',
#                    labels_file='../data/imagelabels.mat',
#                    splits_file='../data/setid.mat')
# train_data, valid_data, test_data = dp.get_data()
# print(train_data.shape, valid_data.shape, test_data.shape)