from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
import math
import torch.utils.data as data
import itertools


#Common method to be used in CambridgeLandmarks and ImagesFromList classes
def load_image(image_path, channels, transform, transform_1d):
    if "rgb" in channels:
        rgb_image = Image.open(image_path)
        if transform != None:
            rgb_image = transform(rgb_image)
        if channels == "rgb":
            return rgb_image

    if "d" in channels:
        segs = image_path.split("/")
        depth_path = "/".join(segs[:-2]) + f"/{segs[-2]}_ZoeDepth/" + segs[-1].split(".")[0] + ".npy"
        depth_image = load_depthmap(depth_path, transform_1d)
        if channels == "d":
            return depth_image

    if channels == "rgbd":
        return  torch.cat((rgb_image, depth_image), dim=0)

def load_depthmap(depth_path, transform_1d):
    data = np.load(depth_path)
    data = np.max(data)-data
    data = (data - np.min(data)) * (255 / (np.max(data) - np.min(data)))
    #data = (data - np.min(data))/ (np.max(data) - np.min(data))
    depth_image = Image.fromarray(data)
    if transform_1d != None:
        depth_image = transform_1d(depth_image)
        return depth_image


class CambridgeLandmarks(Dataset):
    def __init__(self, root_dir, subset='', nNeg=5, transform=None, transform_1d=None, channels="rgb", mode='train',
                        posDistThres=5, negDistThres=6, posAngThres=15, negAngThres=45):

        self.root_dir = root_dir
        self.mode = mode
        self.subsets = [x.strip() for x in subset.split(",")]

        if mode in ["val","eval"]:
            assert len(self.subsets) == 1 ,"Validation supports subset at a time"

        self.transform = transform
        self.transform_1d = transform_1d
        self.channels = channels

        self.posDistThres = posDistThres
        self.negDistThres = negDistThres
        self.posAngThres = posAngThres
        self.negAngThres = negAngThres
        self.nNeg = nNeg

        self.triplets = []
        self.posIndices = {}
        self.negIndices = {}

        self.qData = {}
        self.dbData = {}
        self.qImages = []
        self.dbImages = []


        self.subsetDbs = {}
        for subset in self.subsets:
            startIx = len(self.dbData)
            self.load_metadata(subset)
            endIx = len(self.dbData)
            self.subsetDbs[subset] = list(range(startIx, endIx))

        self.qImages = list(self.qData.keys())
        self.dbImages = list(self.dbData.keys())

        self.qIdx = list(range(len(self.qImages)))
        self.dbIdx = list(range(len(self.dbImages)))
        self.cached_queries = min(1000, len(self.qImages))


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet, target = self.triplets[idx]
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        query = self.__load_image__(os.path.join(self.root_dir, self.qImages[qidx]))
        positive = self.__load_image__(os.path.join(self.root_dir, self.dbImages[pidx]))
        negatives = [self.__load_image__(os.path.join(self.root_dir, self.dbImages[i])) for i in nidx]
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [qidx] #, pidx] + nidx
    

    def load_metadata(self, subset):
        if self.mode == "eval":
            qMetaTxt = os.path.join(self.root_dir, subset, 'dataset_test.txt')
            dbMetaTxt = os.path.join(self.root_dir, subset, 'dataset_train.txt')
            with open(qMetaTxt, "r") as f:
                self.qData = {subset+"/"+r.split()[0]: np.array(r.split()[1:]).astype('float')   for r in f.readlines()[3:]}

            with open(dbMetaTxt, "r") as f:
                self.dbData = {subset+"/"+r.split()[0]: np.array(r.split()[1:]).astype('float')   for r in f.readlines()[3:]}

        if self.mode == "val":
            qMetaTxt = os.path.join(self.root_dir, subset, 'dataset_test.txt')
            dbMetaTxt = os.path.join(self.root_dir, subset, 'dataset_train.txt')
            with open(qMetaTxt, "r") as f:
                metaData = {subset+"/"+r.split()[0]: np.array(r.split()[1:]).astype('float')  for r in f.readlines()[3:]}
                randqIx = np.random.choice(len(metaData), 30, replace=False)
                for i, (k, v) in enumerate(metaData.items()):
                    if i in randqIx:
                        self.qData[k] = v
                        
            with open(dbMetaTxt, "r") as f:
                self.dbData = {subset+"/"+r.split()[0]: np.array(r.split()[1:]).astype('float')   for r in f.readlines()[3:]}
        elif self.mode == "train":
            #during training only use train data 
            metaTxt = os.path.join(self.root_dir, subset, 'dataset_train.txt')
            with open(metaTxt, "r") as f:
                metaData = {subset+"/"+r.split()[0]: np.array(r.split()[1:]).astype('float')   for r in f.readlines()[3:]}
            #choose random queries 30 percent
            randqIx = np.random.choice(len(metaData), 100, replace=False)#int(0.4*len(metaData)), replace=False)
            for i, (k, v) in enumerate(metaData.items()):
                if i in randqIx:
                    self.qData[k] = v
                else:
                    self.dbData[k] = v

        
    def __load_image__(self, image_path):
        return load_image(image_path, self.channels, self.transform, self.transform_1d)


    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices= zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))

        return query, positive, negatives, negCounts, indices

    def get_positives_negatives(self, idx):
        qImage = self.qImages[idx]
        qMeta= self.qData[qImage]

        subset = qImage.split('/')[0]
        dbIdxs = self.subsetDbs[subset]
        results = [self.GetDistanceAndPose(qMeta,self.dbData[self.dbImages[ix]]) for ix in dbIdxs] 
        distances = [r[0] for r in results]
        poses = [r[1] for r in results]
        indexes = np.lexsort((distances, poses))

        pos_idxs = []
        neg_idxs = []
        for i in indexes:
            if distances[i] <= self.posDistThres: #and poses[i] <=self.posAngThres:
                pos_idxs.append(dbIdxs[i])
            if distances[i] >= self.negDistThres:#and poses[i] >= self.negAngThres:
                neg_idxs.append(dbIdxs[i])
        
        if len(pos_idxs) == 0 or len(neg_idxs) == 0:
            #print(idx)
            return False

        self.posIndices[idx] = pos_idxs
        self.negIndices[idx] = neg_idxs
        return True
        
    
    def get_positives_negatives_easy(self, idx):
        qImage = self.qImages[idx]
        qMeta= self.qData[qImage]

        results = [self.GetDistanceAndPose(qMeta,dbMeta) for dbImg, dbMeta in self.dbData.items()] 
        distances = [r[0] for r in results]
        poses = [r[1] for r in results]
        indexes = np.lexsort((distances, poses))

        pos_idxs = []
        neg_idxs = []
        for i in indexes:
            if distances[i] <= self.posDistThres and poses[i] <=self.posAngThres:
                pos_idxs.append(i)
            if distances[i] >= self.negDistThres and poses[i] >= self.negAngThres:
                neg_idxs.append(i)
        
        if len(pos_idxs) == 0 or len(neg_idxs) == 0:
            #print(idx)
            return False

        self.posIndices[idx] = pos_idxs
        self.negIndices[idx] = neg_idxs
        return True

    def GetDistanceAndPose(self, A, B):
        q1 = A[3:] / np.linalg.norm(A[3:])
        q2 = B[3:] /np.linalg.norm(B[3:])
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi
        dist = np.linalg.norm(A[:3]-B[:3])
        return dist, theta


    def update_subcache(self, net=None, pool_size=None):
        self.triplets = []
        qidxs = np.random.choice(len(self.qIdx), self.cached_queries, replace=False)

        for qix in qidxs:
            if qix not in list(self.posIndices.keys()):
                found = self.get_positives_negatives(qix)
                if not found:
                    continue
            #print(qix)

            pidx = np.random.choice(self.posIndices[qix], size=1)[0]
            nidxs = np.random.choice(self.negIndices[qix], size=self.nNeg)
            triplet = [qix, pidx, *nidxs]
            target = [-1, 1] + [0] * len(nidxs)
            self.triplets.append((triplet, target))
        return

    
    def new_epoch(self):
        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qImages) / self.cached_queries)


class ImagesFromList(Dataset):
    def __init__(self, root_dir, subset,  images, transform, transform_1d=None, channels="rgb"):
        self.root_dir = root_dir
        self.subset = subset
        self.images = images
        self.transform = transform
        self.transform_1d = transform_1d
        self.channels = channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = [self.__load_image__(os.path.join(self.root_dir, im)) for im in self.images[idx].split(",")]
        except:
            img = [self.__load_image__(os.path.join(self.root_dir, self.images[idx]))]

        if len(img) == 1:
            img = img[0]

        return img, idx

    def __load_image__(self, image_path):
        return load_image(image_path, self.channels, self.transform, self.transform_1d)

    