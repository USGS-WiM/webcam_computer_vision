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


meta = Meta_getta(0, 0, "Gage_height.ACTIVE_(PRIMARY,_MIXER)@04085108.20190201.csv")

meta.meta_getta("large_dataset/")

img = Meta_getta(sorted(meta.list_obj), 0, 0)

img.img_meta_getta()

Feed_the_mule = Meta_getta(0, 0, 0)

Feed_the_mule.merge_set(img.df_obj, meta.df_obj)


'''

#This part of a group functions involved in extracting meta and target data from a messy data storage system

# As a former scientist I can say that is unrealistic to assume that after 40 years of asking,
# the scientific community will start properly tagging data. 50% of their time is already occupied handleing data.
# The idea for this program is that scientists can just throw their data into a hole and the following tools will
# pull it out and align it for analysis.

#Meta getta is a recusive extraction tool that will mine target data from some point in a file tree. Targets are
#defined by file extention

def meta_getta(path, f_N):
    file_list = glob.glob(path + '**/*.jpg', recursive= True)
    meta = pd.read_csv(path + f_N)
    return file_list, meta

#Thes next few operations are data frame manipulation that will go away with class implementation
img, meta = meta_getta("large_dataset/", "Gage_height.ACTIVE_(PRIMARY,_MIXER)@04085108.20190201.csv")
meta.columns = ['ISO/Time', 'date/time', 'stage reading', 'drp1', 'drp2', 'drp3']


img = sorted(img)
print(len(img))


meta[2] = meta[2].map(str) + ' ' + meta[3]

meta = meta.drop(columns = [3])

meta.columns = ['Orginization', 'Station', 'date/time', 'time zone', 'stage reading', 'disposition']


timestp = []
file_path = []

# When images are without meta, this tool can extract the modification time that is tagged. It future we will want
# hardware to lock, creation time and geocoord to file creation

for i in img:
    mtime = os.stat(i).st_mtime
    timestp.append(int(mtime))
    file_path.append(i)

#Data epoc normilizations for meta
def norm_time(obj):
    new = []
    for i in obj:
        #print (i)
        ep = datetime.datetime.fromtimestamp(i)
        new.append(str(ep.replace(second= 0)))
    return new

def to_epoch(obj, p):
    ep = []
    for i in obj:
        #print(i)
        ep.append(timegm(time.strptime(i, p)))
        tm =timegm(time.strptime(i, p))
        #print(time.strftime(p, time.gmtime(tm)))
    return ep


# More things that go away with class creation
dt = meta['date/time']
dt = dt.tolist()


p = '%m/%d/%Y %H:%M'
a = to_epoch(dt, p)
p = '%Y-%m-%d %H:%M:%S'
b = to_epoch(norm_time(timestp), p)
#print(a)
#print(b)



meta = meta.drop(columns =['ISO/Time', 'drp1', 'drp2', 'drp3'])
meta['date/time'] = a





b = np.array(b)


img_meta = pd.DataFrame(b, columns = ['date/time'])
img_meta['path'] = np.array(file_path)

#print(meta['stage reading'].head(5))
#print(img_meta.head(5))

#img_meta.to_csv("img_check.csv")

# This single line of code is the alignment. It takes all the little adjustments above and does the relational algebra
# to align epochs from pressure plate stage reading to photographs. The next two commands clean up the data
merg = meta.set_index('date/time').join(img_meta.set_index('date/time'))


merg = merg.dropna(axis=0)
merg = merg.reset_index()
#merg.to_csv("checkit.csv")
print(len(merg))

#print(merg[' Value'].head(5), len(merg))
'''
#print(merg.describe)
#print(len(img))
'''

#saves data to .csv in a proper data alignment
merg.to_csv("final_meta.csv")



# This a tool to manually force grayscale on a numpy arrays. Numpy has issues with .pmg images and so this is a
# tool for .png file type. It is unused in this demo
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Takes the feature set aka dependend variables aka X matrix aka the pixel values, converts them to proper dimension,
# crop, and gray scale, then it flattens the matrix of pixels (n x n) into a vector (n^2 x 1). This is added to a
# pre-intantiated array of zeros. Which is need to avoid a BIg (O) of space that is n!
# In other words, that np.zero array is needed or the heap will run out of memory and the kernel will panic.

def get_x(meta):
    x = meta['path']
    path = x.tolist()
    dims = Image.open(path[0])
    w, h = dims.size
    print(h, w, len(meta))
    dimensions = ((2*68)*(2*192))#(int(h-400) * int(w))
    datas = np.zeros(shape=(len(meta['path'].T), dimensions))
    print(datas.shape)
    for i, obj in enumerate(path):
        img = Image.open(obj)
        img= img.convert('L')
        #show_img(img)
        img = resize(img, 0, 0)
        #show_img(img)
        img = np.array(img)
        datas[i] = np.reshape(img, -1)
    return datas


    #print ('\n')
    #for i, F_N in enumerate(path):
        #tmp = (np.asarray(imageio.imread(path)))
        #dat[i] = np.reshape(tmp, -1)
        # This is like some techinacal stuff
        #if i % 100 == 0:
         #   sys.stdout.write(" loading...   %d%%  \r" % (int((i / len(path)) * 100)))
          #  sys.stdout.flush()
            # print (int((i/60327)*100), '%')
    #sys.stdout.write("\033[0;0m")
    #gc.collect()


# Resizes/crops using the janky PIL library

def resize(obj, x, y):
    size = (int(2*192), int(2*108))
    img = obj.resize(size, Image.ANTIALIAS)
    #img = img.filter(ImageFilter.GaussianBlur(radius=2))
    box = (0, int(2*20), int(2*192), int(2*88))
    img= img.crop(box)
    #show_img(img)
    return img

# Will project a image as long as the pixel values don't contain imaginary numbers.

def show_img(obj):
    plt.imshow(obj, cmap='gray')
    plt.show()


# These next few items are for scatch principle component analysis. I don't need it anymore.

def get_mean_vect(dat, depth):
    vect = dat.T
    print(vect.shape)
    mean_vect = []
    for i, obj, in enumerate(depth):
        mean_vect.append(np.mean(vect[i]))
    print('\n--Mean produced--\n')
    return mean_vect

def center_data(dat, dat_mean):
    # print (dat[0][0], "   ", dat_mean[0])
    i = 0
    while i < len(dat):
        dat[i] = dat[i] - dat_mean[i]
        i += 1
    print('\n--Data centered--\n')
    return dat

# Establish Scale

scaler = MinMaxScaler()


# Scale data Function

def scal_it(datas, scal):
    return scal.fit_transform(datas)

'''
# Starting here is our actual CV code

# This is a driver, fetches x for the eigen vectors
# sets the amount of components (ghosts)
# calculates the eigen ghosts and stores them in pca
'''What is happening at this point? Eigen vectors are fragmented versions of some average picture. If my picture is 
100 x 100 pixels then I have 10,000 pixels and therefore I have 10,000 ghosts made up of 10,000 pixels each. If I 
agragate these by adding the values, all fractures become the mean image. The first Eigen ghost has it's pixels 
weighted by what the PCA determined is the most important. In this case we take the first 100, because it reduces 
our dimensions from 10,000 to 100. Much better runtime. The weakest ghost may also be used via boosting but
I didn't do that because adaboost is like terrible to write from scratch and we aren't getting paid for this yet  
'''

'''
def give_up_the_ghosts(xy_train):
    datas = get_x(xy_train)
    n_components = 100
    start = time.time()
    pca = PCA(n_components=n_components, whiten=True).fit(datas)
    end = time.time()
    hacks(start, end)
    return pca

#This is a sudo random value generator to make sure that I actually did something. It also helps be determine
# Out of sample pictures (Person mooning the camera) because if the predicted value is worse than random,
# something is out way out of sample and then I can grab the worst variance from the mean and eject it.
# I haven't written this yet either because I know several people want it and we want to get paid.

def randocalrisian(i):
    s = random.randint(4, i)
    return s-4, s

# This can be used to show a gallery of imaages. It's not essential but good for debugging

def plot_gallery(images, titles, h, w, n_row, n_col):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.5* 4 * n_col, 1.5 * 3 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        ghost = images[i]
        ghost = np.reshape(ghost, (int(680), int(1920)))
        plt.imshow(ghost, cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()
    return


# These runs can take awhile pre-optimization/parrellization, so I wrote a fun alarm that plays random music in 15
# second bursts so I can work on other things while I train a model.

def alert_alarm():
   path = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav", "7.wav", "8.wav", "9.wav", "10.wav", "11.wav"]
   random.shuffle(path)
   playsound("mix_tape/" + path[0])

# This is for time hacks on processes. I don't use it enough.

def hacks(s, e):
    return print("\nTime for process took {:.6f} seconds\n".format(e - s))

# Pulls the stage readings to improve my slope. Converts it into a numpy array for Linear algebra stuff. Magic.

def get_Y(data):
    return np.asarray(data['stage reading'], dtype="|S6")

# Mean squared Error to tell me how dumb this robot is.

def MSE(y_hat, y_true):
    err =np.subtract(y_hat, y_true)
    SE = np.square(err)
    MSE = SE.mean()
    return math.sqrt(MSE)

# Above was the terrible dev process. Below is some optimized code. It does the same basic function but this is
# prime cuts.

def train_model(xy_train, xy_test, Eig):
    x_tr = get_x(xy_train)
    y_tr = get_Y(xy_train)
    x_ts = get_x(xy_test)
    y_ts = get_Y(xy_test)
    start = time.time()
    # apply PCA transformation
    PCA_x_tr = Eig.transform(x_tr)
    PCA_x_ts = Eig.transform(x_ts)
    clf = MLPClassifier(batch_size=150, verbose=True, early_stopping=True).fit(np.array(PCA_x_tr), np.array(y_tr))
    y_pred = clf.predict(PCA_x_ts)
    results = eval_proto(np.asarray(y_pred, dtype=float), np.asarray(y_ts, dtype=float), xy_test)
    mse = MSE(np.asarray(y_pred, dtype=float), np.asarray(y_ts, dtype=float))
    print("True: ", MSE(np.asarray(y_pred, dtype=float), np.asarray(y_ts, dtype=float)))
    #print(classification_report(y_ts, y_pred))

    #a, z = randocalrisian(len(x_ts))
    #y_pred = clf.predict(PCA_x_ts[a:z])
    #print(y_pred, y_ts[a:z])
    #test_it(x_ts[a:z])
    start = time.time()
    print('\n--Reduced dimensions to 9--\n')
    results.to_csv("evaluation_doc.csv", ',')

    return clf, mse

# Ran out of puns at this point.

def test_it(img):
    eigenface_titles = list(range(1, len(img)))
    # ghost = scal_it(K_vect_list, scaler)
    r = len(img)/2
    d = len(img)%2 + r
    plot_gallery(img, eigenface_titles, int(1920), int(680), r, d)

# This is the print out for the results and will one day not be a mess

def eval_proto(yh, yt, df):
    rando = []
    for i in yt:
        rando.append(round(random.uniform(7.10, 11.90), 2))

    print(df.shape)
    print(yt.shape)
    ndf= pd.DataFrame(df['path'])
    ndf['Truth set'] = yt
    ndf['Predicted'] = yh
    ndf['Error'] = np.asarray(abs(yt-yh))
    ndf['Rand_Err'] = np.asarray(abs(yt-rando))
    print("Random: ", MSE(rando, yt))
    ndf.reset_index()
    return ndf.sort_values(['Error'], ascending=0)

# Now starts the final driver code.

def lets_go(Eig, tst, clf):
    x = get_x(tst)
    y= get_Y(tst)
    PCA_x_ts = Eig.transform(x)
    return clf.predict(PCA_x_ts), y


def driver():
    meta = pd.read_csv("final_meta.csv")

    meta, hold_back = train_test_split(meta, test_size=0.001, random_state=random.randrange(843))
    hold_back.to_csv('holdback.txt')
    y_tr= get_Y(hold_back)
    peek(hold_back)
    y_act = y_tr[0]
    goodly = []
    ctr =0
    while len(goodly) < 5:
        ctr+=1
        gc.collect()
        xy_train, xy_test = train_test_split(meta, test_size=0.20, random_state=random.randrange(843))
        Eig = give_up_the_ghosts(xy_train)
        clf, score = train_model(xy_train, xy_test, Eig)
        print(ctr)
        if score < .7:
            y_hat, y_tr = lets_go(Eig, hold_back, clf)
            goodly.append(y_hat[0])
            print("y true = ", y_act, " y hat = ", y_hat[0])
    print (y_act)
    print(KNN_soft(goodly))
    alert_alarm()

# This is like the not actually K nearest Neighbor algo. I will make this better and probably split it from the program
# when we start getting paid.

def KNN_soft(y_hat):
    clss = []
    i = 0
    while i < 201:
        clss.append(round(i*.1, 10))
        i += 1
    KNN =dict.fromkeys(clss, 0)
    for i, obj in enumerate(y_hat):
        tgt = float(y_hat[i])
        check = round(tgt, 1)
        KNN[check]+=1
    best = sorted(((value, key) for (key, value) in KNN.items()), reverse=True)
    return best

# Advance preview of target image. Suprisingly useful

def peek(meta):
    x = meta['path']
    path = x.tolist()
    for i, obj in enumerate(path):
        img = Image.open(obj)
        img = img.convert('L')
        # show_img(img)
        img = resize(img, 0, 0)
        show_img(img)
        break

def KNN_hard():
    print("hello world")


# This is an algo written to enhance night shots. Not in play at this time. It's fast though

def divider(df, seg):
    newdf = pd.DataFrame([[0, 'a', 1, 'a', 1.1, 'a', 'a']], columns = ['date/time', 'Orginization', 'Station', 'time zone', 'stage reading', 'disposition', 'path'])
    for i, obj in enumerate(df.T):
        temp = pd.DataFrame(df[i:i+1])
        path = temp['path'].iloc[0]
        #print(temp.head)
        meta = temp.drop(columns= ['path'])
        #print(meta.head)
        newdf = make_more(newdf, path, meta, seg, i)
    newdf.to_csv('split_meta.csv', ',')



def make_more(df, path, meta, seg, set):
    directory = 'split_cam_img/'
    #print(meta)
    img = Image.open(path)
    rhp, bp = img.size
    lhp = 0
    tp = 0
    sect = int(rhp/seg)
    i = 0
    title = ['lb', 'lub', 'rub', 'rb']
    while i < seg:
        tmp = meta
        box = (lhp, tp, (lhp + sect), bp)
        seg_pic = img.crop(box)
        f_name = directory + 'img' + str(set) + '_' + title[i] +'.jpg'
        seg_pic.save(f_name)
        i += 1
        lhp = lhp + sect + 1
        tmp['path'] = f_name
        print(list(tmp))
        print(list(df))
        df = df.append(tmp, ignore_index=True)
    print(df.head)
    return df
'''

''' ******* COMMAND BLOCK ******* '''
'''
#Produce Eignen Vectors and Values Via PCA

meta = pd.read_csv("final_meta.csv")

xy_train, xy_test = train_test_split(meta, test_size=0.1, random_state=random.randrange(843))



Eig = give_up_the_ghosts(xy_train)
print("\n--PCA complete--\n")

start, end = train_model(xy_train, xy_test, Eig)

driver()
gc.collect()

#save everything here

'''
'''
Tr_file = "Train.npy"
Ts_file = "Test.npy"
Eig_file = "Eig_vect.npy"

#np.save(Tr_file, Tr_datas)
#Tr_DF_h.to_csv("H_TR.csv")
#Ts_DF_h.to_csv("H_TS.csv")
#np.save(Ts_file, Ts_datas)
np.save(Eig_file, K_vect_list)

gc.collect()

#Tr_datas = np.load("Train.npy")
#Ts_datas = np.load("Test.npy")
k_eig = np.load("Eig_vect.npy")
#Tr_DF_h = pd.read_csv("H_TR.csv")
#Ts_DF_h = pd.read_csv("H_TS.csv")
'''


#divider(merg, 4)
