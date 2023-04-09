import numpy as np
import cv2
import pickle
from tqdm import tqdm

__author__ = "Sachin Mehta"

class LoadData:
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        '''
        :param data_dir: directory where the dataset is kept
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        '''
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        '''
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the classWeights variable
        '''
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''
        

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in tqdm(textFile):
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split(',')
                
                img_file = line_arr[0].strip().strip()
                label_file = line_arr[1].strip().strip()

                if trainStg == True:
                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)

                no_files += 1

        
        return 0

    def processData(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')
        return_val = self.readFile('train.txt',True)

        print('Processing validation data')
        return_val1 = self.readFile('val.txt',False)

        print('Pickling data')
        if return_val ==0 and return_val1 ==0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList

            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights

            # pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None




