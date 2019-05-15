import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFilter
import gc
import time
import datetime
from calendar import timegm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import math
from playsound import playsound

#Training moduals
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import collections



#This part of a group functions involved in extracting meta and target data from a messy data storage system

# As a former scientist I can say that is unrealistic to assume that after 40 years of asking,
# the scientific community will start properly tagging data. 50% of their time is already occupied handling data.
# The idea for this program is that scientists can just throw their data into a hole and the following tools will
# pull it out and align it for analysis.



class Meta_getta:

    def __init__(self, list_obj, df_obj, str_obj):
        self.list_obj = list_obj
        self.df_obj = df_obj
        self.str_obj = str_obj

    def meta_getta(self, path):
        self.list_obj = glob.glob(path + '**/*.jpg', recursive= True)
        self.df_obj = pd.read_csv(path + self.str_obj)
        self.df_obj.columns = ['ISO/Time', 'date/time', 'stage reading', 'drp1', 'drp2', 'drp3']
        self.str_obj = '%m/%d/%Y %H:%M'
        temp = self.df_obj['date/time'].values
        self.df_obj = self.df_obj.drop(columns = ['drp1', 'drp2', 'drp3', 'ISO/Time'])
        self.df_obj['date/time'] = np.array(self.to_epoch(temp))


    def img_meta_getta(self):
        temp = []
        #file_path = []
        for i in self.list_obj:
            mtime = os.stat(i).st_mtime
            temp.append(int(mtime))
        self.str_obj = p = '%Y-%m-%d %H:%M:%S'
        self.df_obj = pd.DataFrame(np.array(self.to_epoch(self.norm_time(temp))), columns = ['date/time'])
        self.df_obj['path'] = np.array(self.list_obj)



    def norm_time(self, obj):
        new = []
        for i in obj:
            # print (i)
            ep = datetime.datetime.fromtimestamp(i)
            new.append(str(ep.replace(second=0)))
        return new


    def to_epoch(self, obj):
        ep = []
        for i in obj:
            # print(i)
            ep.append(timegm(time.strptime(i, self.str_obj)))
            tm = timegm(time.strptime(i,self.str_obj))
            # print(time.strftime(p, time.gmtime(tm)))
        return ep

    def merge_set(self, df1, df2):
        self.df_obj = df1.set_index('date/time').join(df2.set_index('date/time'))
        self.df_obj = self.df_obj.dropna(axis=0)
        self.df_obj = self.df_obj.reset_index(drop= True)



##### Command Block Meta Getta #####


meta = Meta_getta(0, 0, "Gage_height.ACTIVE_(PRIMARY,_MIXER)@04085108.20190201.csv")

meta.meta_getta("large_dataset/")

img = Meta_getta(sorted(meta.list_obj), 0, 0)

img.img_meta_getta()

Feed_the_mule = Meta_getta(0, 0, 0)

Feed_the_mule.merge_set(img.df_obj, meta.df_obj)


#saves data to .csv in a proper data alignment
Feed_the_mule.df_obj.to_csv("final_meta.csv")


###### CV block ######





class Cambrain:

    def __init__(self, df_obj, np_obj, size_int, time_strt, time_end):
        self.df_obj = df_obj
        self.np_obj = np_obj
        self.size_int = size_int
        self.time_strt = time_strt
        self.time_end = time_end

    def hacks(self):
        return "\nTime for process took {:.6f} seconds\n".format(self.time_end - self.time_strt)

    def give_up_the_ghosts(self):
        n_components = self.np_obj
        self.time_strt = time.time()
        self.np_obj = SKPCA(n_components=n_components, whiten=True).fit(self.df_obj)
        self.time_end = time.time()
        print(self.hacks())

    # Resizes/crops using the janky PIL library

    def re_size(self):
        size = (int(self.size_int * 192), int(self.size_int * 108))
        img = self.df_obj.resize(size, Image.ANTIALIAS)
        box = (0, int(self.size_int * 20), int(self.size_int * 192), int(self.size_int * 88))
        self.df_obj = img.crop(box)


    # Will project a image as long as the pixel values don't contain imaginary numbers.


    def get_x(self):
        self.time_strt = time.time()
        x = self.df_obj['path']
        path = x.tolist()
        dims = Image.open(path[0])
        w, h = dims.size
        #print(h, w, len(meta))
        dimensions = ((self.size_int * 68) * (self.size_int * 192))  # (int(h-400) * int(w))
        datas = np.zeros(shape=(len(self.df_obj['path'].values), dimensions))
        #print(datas.shape)
        for i, obj in enumerate(path):
            self.df_obj = Image.open(obj)
            self.df_obj = self.df_obj.convert('L')
            self.re_size()
            self.df_obj = np.array(self.df_obj)
            datas[i] = np.reshape(self.df_obj, -1)
        self.df_obj = datas
        self.time_end = time.time()
        self.hacks()

    def get_Y(self):
        self.np_obj = np.asarray(self.df_obj['stage reading'], dtype="|S6")



class CAMBOT():

    def __init__(self, TR, TS, LS, HB, PCA, HARD, SOFT, Y_pred, int_obj, target_obj):
        self.TR = TR
        self.TS = TS
        self.PCA = PCA
        self.LS = LS
        self.HB = HB
        self.HARD = HARD
        self.SOFT = SOFT
        self.Y_pred = Y_pred
        self.int_obj = int_obj
        self.target_obj = target_obj
        self.RMSE = 0
        self.randy = 0

    def train_model(self):
        self.LS.time_strt = time.time()
        self.TR.get_Y()
        self.TR.get_x()
        self.TS.get_Y()
        self.TS.get_x()
        self.PCA.df_obj = self.TR.df_obj
        self.PCA.np_obj = self.int_obj
        self.PCA.give_up_the_ghosts()
        PCA_x_tr = self.PCA.np_obj.transform(self.TR.df_obj)
        PCA_x_ts = self.PCA.np_obj.transform(self.TS.df_obj)
        clf = MLPClassifier(batch_size=150, verbose=True, early_stopping=True).fit(np.array(PCA_x_tr), np.array(self.TR.np_obj))
        self.Y_pred = clf.predict(PCA_x_ts)
        print('\n--Reduced dimensions to 9--\n')
        #results.to_csv("evaluation_doc.csv", ',')
        self.LS.time_end = time.time()
        self.LS.hacks()
        return clf

    def GO(self):
        self.target_obj = self.HB.df_obj
        #hold_back.to_csv('holdback.txt')
        self.peek()
        self.HB.get_Y()
        self.HB.get_x()
        self.SOFT.pred_obj = []
        ctr = 0
        while len(self.SOFT.pred_obj) < 100:
            ctr += 1
            gc.collect()
            self.TR.df_obj, self.TS.df_obj = train_test_split(self.LS.df_obj, test_size=0.20, random_state=random.randrange(843))
            clf = self.train_model()
            self.MSE()
            print('--Model Accuracy--\n\nRandom prediction: ', self.randy, '\nAlgorthim prediction: ', self.RMSE, '\nThreshold: 0.7')
            print('Total Runs: ', ctr)
            if self.RMSE < .7:
                PCA_x_ts = self.PCA.np_obj.transform(self.HB.df_obj)
                self.Y_pred = clf.predict(PCA_x_ts)
                self.SOFT.pred_obj.append(self.Y_pred[0])
                print("Successful tests: ", len(self.SOFT.pred_obj))
                print("\n--Viable slope--\n\ny true = ", self.HB.np_obj[0].decode('utf-8'), " ft   y predicted = ", self.Y_pred[0].decode('utf-8'), 'ft\n')
        self.HARD.pred_obj = self.SOFT.pred_obj
        self.SOFT.np_obj = self.HB.np_obj[0]
        self.HARD.np_obj = self.HB.np_obj[0]
        self.SOFT.VERIFY()
        self.HARD.VERIFY()
        #print ('\nHard Precision: ', 100*(self.HARD.FP/len(self.SOFT.pred_obj)), '%%\nSoft Prediction: ', 100 * (self.SOFT.FP/len(self.SOFT.pred_obj)),
        #       '%%\n\nConfidence score: ')
        #alert_alarm()

    # Mean squared Error to tell me how dumb this robot is.

    def MSE(self):
        self.randy = [np.array(self.TS.np_obj, dtype = float).mean()] * len(self.TS.np_obj)
        for i, obj in enumerate(self.randy):
            self.randy[i] = obj*self.randocalrisian()
        err = np.subtract(np.array(self.randy, dtype=float), np.array(self.TS.np_obj, dtype=float))
        SE = np.square(err)
        self.randy = math.sqrt(SE.mean())
        err = np.subtract(np.array(self.Y_pred, dtype=float), np.array(self.TS.np_obj, dtype=float))
        SE = np.square(err)
        self.RMSE = math.sqrt(SE.mean())

    def randocalrisian(self):
        s = random.uniform(.25, 1.75)
        return s


    def peek(self):
        x = self.target_obj['path']
        path = x.tolist()
        for i, obj in enumerate(path):
            img = Image.open(obj)
            img = img.resize((3*192, 3*108))
            plt.imshow(img)
            plt.show()
            break

class verify:

    def __init__(self, acc_obj, depth):
        self.pred_obj = 0
        self.FP = 0
        self.acc_obj = acc_obj
        self.TP = 0
        self.np_obj = 0
        self.depth = depth
        d = decimal.Decimal(str(acc_obj))
        r = d.as_tuple().exponent
        self.precision = abs(r)
        self.rng = depth * (10**self.precision)
        print(self.rng)

    def VERIFY(self):
        clss = []
        i = 0
        while i < self.rng+1:
            clss.append(round(i * self.acc_obj, self.precision))
            i += 1
        KNN = dict.fromkeys(clss, 0)
        for i, obj in enumerate(self.pred_obj):
            tgt = float(self.pred_obj[i])
            check = round(tgt, self.precision)
            KNN[check] += 1
        best = sorted(((value, key) for (key, value) in KNN.items()), reverse=True)
        choice = best[0][1]
        TP = best[0][0]
        Precision = (TP/len(self.pred_obj)) * 100
        print (choice, '\n', TP, '\n', Precision, '%')
        FP = np.subtract(np.array(best), np.array((0, best[0][1])))
        print(FP)
        a = np.array()
        print(np.where(best[:100,][1] > 0, a, best[:100,][1]))






##### Command Block #####

Hold_back = Cambrain(0, 0, 2, 0, 0)

Learning_set = Cambrain(0, 0, 2, 0, 0)

#Produce Eignen Vectors and Values Via PCA

datas = pd.read_csv("final_meta.csv")

Learning_set.df_obj, Hold_back.df_obj = train_test_split(datas, test_size=0.001, random_state=random.randrange(843))

print (Learning_set.df_obj.shape)

PCA = Cambrain(Learning_set.df_obj, 0, 2, 0, 2)

Tr_set = Cambrain(Learning_set.df_obj ,0 ,2 ,0 ,0)

Ts_set = Cambrain(Learning_set.df_obj, 0, 2, 0, 0)

Target_set = Cambrain(Hold_back.df_obj, 0, 2, 0, 0)

soft = verify(0.1, 20)

hard = verify(0.01, 20)

cambot = CAMBOT(Tr_set, Ts_set, Learning_set, Hold_back, PCA, hard, soft, 0, 100, 0)

cambot.GO()



