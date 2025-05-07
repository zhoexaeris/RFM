import torchvision
import os
import utils.DataTools as dt

# Keep the same augmentation transforms as the original
aug_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

aug_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

class CustomDataset:
    def __init__(self, folder_path="./dataset_root"):
        self.folder_path = folder_path
        self.R_dir = ["real"]  # Real images directory name
        self.F_dir = ["fake"]  # Fake images directory name
        
        # Define path functions for each split
        self.trainpath = lambda path, file: os.path.join(self.folder_path, "train", file)
        self.validpath = lambda path, file: os.path.join(self.folder_path, "validate", file)
        self.testpath = lambda path, file: os.path.join(self.folder_path, "test", file)

    def getDatasets(self, pathfunc, infolist, transform, process=None, datasetfunc=None):
        datalist = []
        for info in infolist:
            discribe = info[0]
            dirlist = info[1]
            label = info[2]
            cnt = 0
            for dirname in dirlist:
                path = pathfunc(self.folder_path, dirname)
                cnt += len(os.listdir(path))
                datalist.append((path, label))
            print(discribe, cnt)
        if datasetfunc is not None:
            dataset = datasetfunc(datalist, transform=transform, process=process)
        else:
            dataset = dt.imgdataset(datalist, transform=transform, process=process)
        return dataset

    def getsetlist(self, real, setType, process=None, datasetfunc=None):
        setdir = self.R_dir if real is True else self.F_dir
        label = 0 if real is True else 1
        aug = aug_train if setType == 0 else aug_test
        pathfunc = self.trainpath if setType == 0 else self.validpath if setType == 1 else self.testpath
        setlist = []
        for setname in setdir:
            datalist = [(pathfunc(self.folder_path, setname), label)]
            if datasetfunc is not None:
                tmptestset = datasetfunc(datalist, transform=aug, process=process)
            else:
                tmptestset = dt.imgdataset(datalist, transform=aug, process=process)
            setlist.append(tmptestset)
        return setlist, setdir

    def getTrainsetR(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath, [[self.__class__.__name__+" TrainsetR", self.R_dir, 0]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath, [[self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainset(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath, [[self.__class__.__name__+" TrainsetR", self.R_dir, 0], [self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getValidsetR(self, process=None, datasetfunc=None):
        return self.getDatasets(self.validpath, [[self.__class__.__name__+" ValidsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidsetF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.validpath, [[self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidset(self, process=None, datasetfunc=None):
        return self.getDatasets(self.validpath, [[self.__class__.__name__+" ValidsetR", self.R_dir, 0], [self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetR(self, process=None, datasetfunc=None):
        return self.getDatasets(self.testpath, [[self.__class__.__name__+" TestsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.testpath, [[self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestset(self, process=None, datasetfunc=None):
        return self.getDatasets(self.testpath, [[self.__class__.__name__+" TestsetR", self.R_dir, 0], [self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc) 