import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#Import the right package
from scipy.io import loadmat
from PIL import Image
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio
import sys
import gc
import math
import random


# Data Load Function

def get_data(path, dat, folder):
    print ('\n')
    sys.stdout.write("\033[1;31m")
    for i, F_N in enumerate(path):
        tmp = (np.asarray(imageio.imread(folder + F_N)))
        dat[i] = np.reshape(tmp, -1)
        # This is like some techinacal stuff
        if i % 100 == 0:
            sys.stdout.write(" loading...   %d%%  \r" % (int((i / len(path)) * 100)))
            sys.stdout.flush()
            # print (int((i/60327)*100), '%')
    sys.stdout.write("\033[0;0m")
    gc.collect()
    return (dat)


# Mean Calculation Function

def get_mean_vect(dat, depth):
    vect = dat.T
    mean_vect = []
    for i, obj, in enumerate(depth):
        # print(vect[i])
        mean_vect.append(np.mean(vect[i]))
    return mean_vect


# Data centering function

def center_data(dat, dat_mean):
    # print (dat[0][0], "   ", dat_mean[0])
    i = 0
    while i < 1000:
        dat[i] = dat[i] - dat_mean[i]
        i += 1
    return dat


# Establish Scale

scaler = MinMaxScaler()


# Scale data Function

def scal_it(datas, scal):
    return scal.fit_transform(datas)


# Projection Function

def projection(A_tr, Eig_vect):
    return A_tr.dot(Eig_vect)


# Shuffler on batch selection Function

def shuffle_dat(dat, headers, rng):
    idx = random.randrange(len(dat))
    if idx > rng:
        idx -= rng
    return np.asarray(dat[idx:idx + rng]), headers[idx:idx + rng]


# Gradient calculator Function

def cost_gradient(x, w, age):
    return np.asarray(w.dot(x) - age)


# Stocastic Gradient Decent Function

def SGD_tr(Tr, ep, age, alpha, bias, rng):
    # w = np.random.rand(20, 1)
    # print (Sub_proj.shape)
    Sub_proj, sub_age = shuffle_dat(Tr, age, rng)
    Sub_proj = np.concatenate((Sub_proj, bias), axis=1)
    print (Sub_proj.shape)
    w = Sub_proj[0]
    # print (w.shape)
    while ep > 0:
        gc.collect()
        for j, obj, in enumerate(Sub_proj):
            grad = cost_gradient(Sub_proj[j], w, sub_age[j])
            grad = np.asarray(grad.dot(Sub_proj[j]))
            grad = alpha.dot(grad)
            w = w - grad
            # print(w)
        Sub_proj, sub_age = shuffle_dat(Tr, age, rng)
        # print (Sub_proj.shape)
        Sub_proj = np.concatenate((Sub_proj, bias), axis=1)
        ep -= 1
    return w


# Prep Test Set Function

def prep_ts(Ts_dat, K_eig, cols):
    mean_vect = get_mean_vect(Ts_dat, cols)
    datas = center_data(Ts_dat, mean_vect)
    datas = scaler.transform(datas)

    return projection(datas, K_eig.T)


# Mean squared error function

def MSE(y, yp, label):
    print("\n***%s***" % label)
    Err = y - yp
    SE = Err ** 2
    return (SE.mean())


def plot_gallery(images, titles, h, w, n_row, n_col):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        ghost = np.reshape(images[i], (h, w))
        plt.imshow(ghost, cmap='gray')
        # plt.title(titles[i], size=len(tiles))
        plt.xticks(())
        plt.yticks(())


def main(k_n):
    sys.stdout.write("\033[0;0m")

    # Load the data
    metaData = loadmat("wiki_labeled.mat", squeeze_me=True, struct_as_record=True)

    # Get the dob, ID, Path field from the structure data
    age = np.asarray(metaData['wiki_labeled']['age'].item(0))
    ID = np.asarray(metaData['wiki_labeled']['ID'].item(0))
    f_name = metaData['wiki_labeled']['full_path'].item(0)

    # build a splitable dataframe

    split = pd.DataFrame([ID, age, f_name])
    split = split.T.rename(columns={0: 'ID', 1: 'age', 2: 'Path'})

    # Clean out erronous out of sample data that cannot be correct (Age = 0)

    split = split[split.age != 0]

    # Split it on a random seed
    (Datas_Tr, Datas_Ts) = train_test_split(split, test_size=0.2, random_state=random.randrange(9000))

    # Transform everthing to the correct dimensions

    Tr_DF_h = pd.DataFrame([Datas_Tr['ID'], Datas_Tr['age']])
    Ts_DF_h = pd.DataFrame([Datas_Ts['ID'], Datas_Ts['age']])

    # Headers for later aka the y's

    Tr_DF_h = Tr_DF_h.T
    Ts_DF_h = Ts_DF_h.T

    # Instantiate a dataset to make everything fast and .good()

    datas = np.zeros(shape=(len(Datas_Tr['Path'].T), 10000))

    # x_plot= range(len(datas))

    # Load the x_datas

    Tr_datas = (get_data(Datas_Tr['Path'].values, datas, 'wiki_labeled/'))
    datas = np.zeros(shape=(len(Datas_Ts['Path'].T), 10000))
    Ts_datas = (get_data(Datas_Ts['Path'].values, datas, 'wiki_labeled/'))

    print ("\nData Loaded\n")

    # enermeration object the lazy way

    cols = [0] * 10000

    # Call mean calculation on Training set

    mean_vect = get_mean_vect(Tr_datas, cols)
    gc.collect()

    print ("\nMean calculated\n")

    # Center data

    datas = center_data(Tr_datas, mean_vect)
    gc.collect()

    print ("\nData Centered\n")

    # Save Training Data for later steps

    Tr_datas = datas.T

    # Start the PCA here

    gc.collect()
    MbyM = (datas.T.dot(datas)) / (len(datas.T) - 1)
    gc.collect()

    print ("\nCovariences calculated\n")

    # compute the eigen values

    Eigen_vals, Eigen_vect = LA.eig(MbyM)
    gc.collect()

    # sort eigens
    idx = Eigen_vals.argsort()[::-1]
    Eigen_vals = Eigen_vals[idx]
    Eigen_vect = Eigen_vect[:, idx]

    gc.collect()

    # Select K_size and split

    K_vals_list = Eigen_vals[:K_n]
    K_vect_list = Eigen_vect.T[:K_n]

    # scree plot

    x_plt = list(range(len(K_vals_list)))

    fig = plt.figure(figsize=(8, 5))
    # sing_vals = np.arange(num_vars) + 1
    plt.plot(x_plt, K_vals_list, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')

    leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    plt.show()

    # Reconstruct Eigens
    eigenface_titles = list(range(1, 20))
    print(len(eigenface_titles))
    # ghost = np.reshape(i, (100, 100))
    plot_gallery(K_vect_list, eigenface_titles, 100, 100, 4, 5)
    '''
    for i in K_vect_list:
        ghost = np.reshape(i,(100, 100))
        plt.imshow(ghost, cmap='gray')
        plt.show()



    ghost = np.reshape(mean_vect,(100, 100))
    plt.imshow(ghost, cmap='gray')
'''
    plt.show()

    # save everything here

    # Scale data

    Tr_file = "Train.npy"
    Ts_file = "Test.npy"
    Eig_file = "Eig_vect.npy"

    np.save(Tr_file, Tr_datas)
    Tr_DF_h.to_csv("H_TR.csv")
    Ts_DF_h.to_csv("H_TS.csv")
    np.save(Ts_file, Ts_datas)
    np.save(Eig_file, K_vect_list)

    gc.collect()

    Tr_datas = np.load("Train.npy")
    Ts_datas = np.load("Test.npy")
    K_eig = np.load("Eig_vect.npy")
    Tr_DF_h = pd.read_csv("H_TR.csv")
    Ts_DF_h = pd.read_csv("H_TS.csv")

    K_eig = K_vect_list

    # Projectection

    Projected_tr = projection(scal_it(Tr_datas.T, scaler), K_eig.T)

    # Variables for SVD

    batch_size = 20
    epoch = 12000
    alpha = .00002
    alpha = np.asarray(alpha)
    bias = np.ones((batch_size, 1), dtype=np.int)

    # Produce Slope

    print ("\nBegin SGD\n")

    w = SGD_tr(Projected_tr, epoch, Tr_DF_h['age'].values, alpha, bias, batch_size)

    best_w = "best_w.npy"

    # np.save(best_w, w)

    print ("\nw produced\n")

    # Get the Ts set ready

    proj_ts = prep_ts(Ts_datas, K_eig, cols)
    bias = np.ones((len(Ts_datas), 1), dtype=np.int)
    proj_ts = np.concatenate((proj_ts, bias), axis=1)

    # Get Y_hat and Y_true
    y_hat = proj_ts.dot(w)
    y_true = Ts_DF_h['age']

    print("Epoch: ", epoch)
    print("Alpha: ", ".00002")
    print("Batch Size: ", batch_size)

    # RMSE
    return (math.sqrt(MSE(y_true, y_hat, "RMSE")))


# Data Load Function

def get_data(path, dat, folder):
    print ('\n')
    sys.stdout.write("\033[1;31m")
    for i, F_N in enumerate(path):
        tmp = (np.asarray(imageio.imread(folder + F_N)))
        dat[i] = np.reshape(tmp, -1)
        # This is like some techinacal stuff
        if i % 100 == 0:
            sys.stdout.write(" loading...   %d%%  \r" % (int((i / len(path)) * 100)))
            sys.stdout.flush()
            # print (int((i/60327)*100), '%')
    sys.stdout.write("\033[0;0m")
    gc.collect()
    return (dat)


# Mean Calculation Function

def get_mean_vect(dat, depth):
    vect = dat.T
    mean_vect = []
    for i, obj, in enumerate(depth):
        # print(vect[i])
        mean_vect.append(np.mean(vect[i]))
    return mean_vect


# Data centering function

def center_data(dat, dat_mean):
    # print (dat[0][0], "   ", dat_mean[0])
    i = 0
    while i < 1000:
        dat[i] = dat[i] - dat_mean[i]
        i += 1
    return dat


# Establish Scale

scaler = MinMaxScaler()


# Scale data Function

def scal_it(datas, scal):
    return scal.fit_transform(datas)


# Projection Function

def projection(A_tr, Eig_vect):
    return A_tr.dot(Eig_vect)


# Shuffler on batch selection Function

def shuffle_dat(dat, headers, rng):
    idx = random.randrange(len(dat))
    if idx > rng:
        idx -= rng
    return np.asarray(dat[idx:idx + rng]), headers[idx:idx + rng]


# Gradient calculator Function

def cost_gradient(x, w, age):
    return np.asarray(w.dot(x) - age)


# Stocastic Gradient Decent Function

def SGD_tr(Tr, ep, age, alpha, bias, rng):
    # w = np.random.rand(20, 1)
    # print (Sub_proj.shape)
    Sub_proj, sub_age = shuffle_dat(Tr, age, rng)
    Sub_proj = np.concatenate((Sub_proj, bias), axis=1)
    print (Sub_proj.shape)
    w = Sub_proj[0]
    # print (w.shape)
    while ep > 0:
        gc.collect()
        for j, obj, in enumerate(Sub_proj):
            grad = cost_gradient(Sub_proj[j], w, sub_age[j])
            grad = np.asarray(grad.dot(Sub_proj[j]))
            grad = alpha.dot(grad)
            w = w - grad
            # print(w)
        Sub_proj, sub_age = shuffle_dat(Tr, age, rng)
        # print (Sub_proj.shape)
        Sub_proj = np.concatenate((Sub_proj, bias), axis=1)
        ep -= 1
    return w


# Prep Test Set Function

def prep_ts(Ts_dat, K_eig, cols):
    mean_vect = get_mean_vect(Ts_dat, cols)
    datas = center_data(Ts_dat, mean_vect)
    datas = scaler.transform(datas)

    return projection(datas, K_eig.T)


# Mean squared error function

def MSE(y, yp, label):
    print("\n***%s***" % label)
    Err = y - yp
    SE = Err ** 2
    return (SE.mean())


def plot_gallery(images, titles, h, w, n_row, n_col):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        ghost = np.reshape(images[i], (h, w))
        plt.imshow(ghost, cmap='gray')
        # plt.title(titles[i], size=len(tiles))
        plt.xticks(())
        plt.yticks(())


def main(k_n):
    sys.stdout.write("\033[0;0m")

    # Load the data
    metaData = loadmat("wiki_labeled.mat", squeeze_me=True, struct_as_record=True)

    # Get the dob, ID, Path field from the structure data
    age = np.asarray(metaData['wiki_labeled']['age'].item(0))
    ID = np.asarray(metaData['wiki_labeled']['ID'].item(0))
    f_name = metaData['wiki_labeled']['full_path'].item(0)

    # build a splitable dataframe

    split = pd.DataFrame([ID, age, f_name])
    split = split.T.rename(columns={0: 'ID', 1: 'age', 2: 'Path'})

    # Clean out erronous out of sample data that cannot be correct (Age = 0)

    split = split[split.age != 0]

    # Split it on a random seed
    (Datas_Tr, Datas_Ts) = train_test_split(split, test_size=0.2, random_state=random.randrange(9000))

    # Transform everthing to the correct dimensions

    Tr_DF_h = pd.DataFrame([Datas_Tr['ID'], Datas_Tr['age']])
    Ts_DF_h = pd.DataFrame([Datas_Ts['ID'], Datas_Ts['age']])

    # Headers for later aka the y's

    Tr_DF_h = Tr_DF_h.T
    Ts_DF_h = Ts_DF_h.T

    # Instantiate a dataset to make everything fast and .good()

    datas = np.zeros(shape=(len(Datas_Tr['Path'].T), 10000))

    # x_plot= range(len(datas))

    # Load the x_datas

    Tr_datas = (get_data(Datas_Tr['Path'].values, datas, 'wiki_labeled/'))
    datas = np.zeros(shape=(len(Datas_Ts['Path'].T), 10000))
    Ts_datas = (get_data(Datas_Ts['Path'].values, datas, 'wiki_labeled/'))

    print ("\nData Loaded\n")

    # enermeration object the lazy way

    cols = [0] * 10000

    # Call mean calculation on Training set

    mean_vect = get_mean_vect(Tr_datas, cols)
    gc.collect()

    print ("\nMean calculated\n")

    # Center data

    datas = center_data(Tr_datas, mean_vect)
    gc.collect()

    print ("\nData Centered\n")

    # Save Training Data for later steps

    Tr_datas = datas.T

    # Start the PCA here

    gc.collect()
    MbyM = (datas.T.dot(datas)) / (len(datas.T) - 1)
    gc.collect()

    print ("\nCovariences calculated\n")

    # compute the eigen values

    Eigen_vals, Eigen_vect = LA.eig(MbyM)
    gc.collect()

    # sort eigens
    idx = Eigen_vals.argsort()[::-1]
    Eigen_vals = Eigen_vals[idx]
    Eigen_vect = Eigen_vect[:, idx]

    gc.collect()

    # Select K_size and split

    K_vals_list = Eigen_vals[:K_n]
    K_vect_list = Eigen_vect.T[:K_n]

    # scree plot

    x_plt = list(range(len(K_vals_list)))

    fig = plt.figure(figsize=(8, 5))
    # sing_vals = np.arange(num_vars) + 1
    plt.plot(x_plt, K_vals_list, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')

    leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    plt.show()

    # Reconstruct Eigens
    eigenface_titles = list(range(1, 20))
    print(len(eigenface_titles))
    # ghost = np.reshape(i, (100, 100))
    plot_gallery(K_vect_list, eigenface_titles, 100, 100, 4, 5)
    '''
    for i in K_vect_list:
        ghost = np.reshape(i,(100, 100))
        plt.imshow(ghost, cmap='gray')
        plt.show()



    ghost = np.reshape(mean_vect,(100, 100))
    plt.imshow(ghost, cmap='gray')
'''
    plt.show()

    # save everything here

    # Scale data

    Tr_file = "Train.npy"
    Ts_file = "Test.npy"
    Eig_file = "Eig_vect.npy"

    np.save(Tr_file, Tr_datas)
    Tr_DF_h.to_csv("H_TR.csv")
    Ts_DF_h.to_csv("H_TS.csv")
    np.save(Ts_file, Ts_datas)
    np.save(Eig_file, K_vect_list)

    gc.collect()

    Tr_datas = np.load("Train.npy")
    Ts_datas = np.load("Test.npy")
    K_eig = np.load("Eig_vect.npy")
    Tr_DF_h = pd.read_csv("H_TR.csv")
    Ts_DF_h = pd.read_csv("H_TS.csv")

    K_eig = K_vect_list

    # Projectection

    Projected_tr = projection(scal_it(Tr_datas.T, scaler), K_eig.T)

    # Variables for SVD

    batch_size = 20
    epoch = 12000
    alpha = .00002
    alpha = np.asarray(alpha)
    bias = np.ones((batch_size, 1), dtype=np.int)

    # Produce Slope

    print ("\nBegin SGD\n")

    w = SGD_tr(Projected_tr, epoch, Tr_DF_h['age'].values, alpha, bias, batch_size)

    best_w = "best_w.npy"

    # np.save(best_w, w)

    print ("\nw produced\n")

    # Get the Ts set ready

    proj_ts = prep_ts(Ts_datas, K_eig, cols)
    bias = np.ones((len(Ts_datas), 1), dtype=np.int)
    proj_ts = np.concatenate((proj_ts, bias), axis=1)

    # Get Y_hat and Y_true
    y_hat = proj_ts.dot(w)
    y_true = Ts_DF_h['age']

    print("Epoch: ", epoch)
    print("Alpha: ", ".00002")
    print("Batch Size: ", batch_size)

    # RMSE
    return (math.sqrt(MSE(y_true, y_hat, "RMSE")))
