

import glob
from collections import defaultdict
import os
import numpy as np
import random


import torchvision.transforms as transforms

from PIL import Image
from skimage import io

def default_loader(path):
    return Image.open(path).convert('RGB')


def eurosat_loader(path):
    return io.imread(path)

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.eye(n_classes)[x]

class DataGeneratorSplitting:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            # train_subdirImgPth = subdirImgPth[:int(0.2*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.2*len(subdirImgPth)):int(0.3*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):]
            
            train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)



class DataGeneratorTiplet:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.train_label2idx = defaultdict()
        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.labels_list = None

        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)
            
            train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]
            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)

                if label in self.train_label2idx:
                    self.train_label2idx[label].append(train_count)
                else:
                    self.train_label2idx[label] = [train_count]
                train_count += 1
                
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        self.labels_list = list(range(len(self.sceneList)))

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))


    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
            _, imgLb = self.train_idx2fileDict[idx]
            positive_index = idx
            while positive_index == idx:
                positive_index = np.random.choice(self.train_label2idx[imgLb])
            
            negative_label = np.random.choice(list(set(self.labels_list) - set([imgLb])))
            negative_index = np.random.choice(self.train_label2idx[negative_label])
            return self.__data_generation_triplet(idx, positive_index, negative_index)

        elif self.phase == 'val':
            idx = self.valDataIndex[index]
            return self.__data_generation(idx)
        else:
            idx = self.testDataIndex[index]
            return self.__data_generation(idx)


    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]
        if self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)

        # return {'img': img, 'label': imgLb, 'idx':idx}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb}

    def __data_generation_triplet(self, idx, pos_idx, neg_idx):

        anc_imgPth, anc_label = self.train_idx2fileDict[idx]
        pos_imgPth, _ = self.train_idx2fileDict[pos_idx]
        neg_imgPth, _ = self.train_idx2fileDict[neg_idx]

        anc_img = default_loader(anc_imgPth)
        pos_img = default_loader(pos_imgPth)
        neg_img = default_loader(neg_imgPth)

        if self.imgTransform is not None:
            anc_img = self.imgTransform(anc_img)
            pos_img = self.imgTransform(pos_img)
            neg_img = self.imgTransform(neg_img)

        return {'anc':anc_img, 'pos':pos_img, 'neg':neg_img, 'anc_label':anc_label}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)


class DataGenClsSplit:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.clsNum = len(self.sceneList)

        self.trainCls = None
        self.testCls = None

        self.train_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.test_idx2fileDict = defaultdict()

        self.train_scene2Label = defaultdict()
        self.test_scene2Label = defaultdict()

        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        random.seed(42)
        self.trainCls = random.sample(self.sceneList, int(self.clsNum*0.8))
        self.testCls = list(set(self.sceneList) - set(self.trainCls))

        self.train_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.trainCls):
            self.train_scene2Label[os.path.basename(scenePth)] = label
            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth[:int(0.8*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            self.train_numImgs += len(train_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1

        print("total number of seen classes: {}".format(len(self.trainCls)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))

        self.test_numImgs = 0
        test_count = 0

        for label, scenePth in enumerate(self.testCls):
            self.test_scene2Label[os.path.basename(scenePth)] = label
            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.shuffle(subdirImgPth)

            self.test_numImgs += len(subdirImgPth)

            for imgPth in subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1

        print("total number of unseen classes: {}".format(len(self.testCls)))
        print("total number of test images: {}".format(self.test_numImgs))

        self.trainDataIndex = list(range(self.train_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

        self.testDataIndex = list(range(self.test_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)


    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}


    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)


class DataGenClsSplitTriplet:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.clsNum = len(self.sceneList)

        self.trainCls = None
        self.testCls = None

        self.train_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.test_idx2fileDict = defaultdict()

        self.train_label2idx = defaultdict()
        
        self.train_scene2Label = defaultdict()
        self.test_scene2Label = defaultdict()

        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.labels_list = None

        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        random.seed(42)
        self.trainCls = random.sample(self.sceneList, int(self.clsNum*0.8))
        self.testCls = list(set(self.sceneList) - set(self.trainCls))

        self.train_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.trainCls):
            self.train_scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth[:int(0.8*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            self.train_numImgs += len(train_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)

                if label in self.train_label2idx:
                    self.train_label2idx[label].append(train_count)
                else:
                    self.train_label2idx[label] = [train_count]
                train_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        self.labels_list = list(range(len(self.trainCls)))

        print("total number of seen classes: {}".format(len(self.trainCls)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))

        self.test_numImgs = 0
        test_count = 0

        for label, scenePth in enumerate(self.testCls):
            self.test_scene2Label[os.path.basename(scenePth)] = label
            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.shuffle(subdirImgPth)

            self.test_numImgs += len(subdirImgPth)

            for imgPth in subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1

        print("total number of unseen classes: {}".format(len(self.testCls)))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))


    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
            _, imgLb = self.train_idx2fileDict[idx]
            positive_index = idx
            while positive_index == idx:
                positive_index = np.random.choice(self.train_label2idx[imgLb])
            
            negative_label = np.random.choice(list(set(self.labels_list) - set([imgLb])))
            negative_index = np.random.choice(self.train_label2idx[negative_label])
            return self.__data_generation_triplet(idx, positive_index, negative_index)

        elif self.phase == 'val':
            idx = self.valDataIndex[index]
            return self.__data_generation(idx)
        else:
            idx = self.testDataIndex[index]
            return self.__data_generation(idx)


    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]
        if self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)

        # return {'img': img, 'label': imgLb, 'idx':idx}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb}

    def __data_generation_triplet(self, idx, pos_idx, neg_idx):

        anc_imgPth, anc_label = self.train_idx2fileDict[idx]
        pos_imgPth, _ = self.train_idx2fileDict[pos_idx]
        neg_imgPth, _ = self.train_idx2fileDict[neg_idx]

        anc_img = default_loader(anc_imgPth)
        pos_img = default_loader(pos_imgPth)
        neg_img = default_loader(neg_imgPth)

        if self.imgTransform is not None:
            anc_img = self.imgTransform(anc_img)
            pos_img = self.imgTransform(pos_img)
            neg_img = self.imgTransform(neg_img)

        return {'anc':anc_img, 'pos':pos_img, 'neg':neg_img, 'anc_label':anc_label}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)





if __name__ == "__main__":
    
    datagen = DataGeneratorSplitting(
        data='/home/jkang/Documents/data',
        # dataset='AID'
        dataset='NWPU-RESISC45'
    )

    print(list(map(os.path.basename, datagen.sceneList)))







