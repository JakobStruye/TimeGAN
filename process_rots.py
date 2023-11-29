import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, ccf
import scipy
np.set_printoptions(suppress=True)
plt.rcParams.update({'font.size': 18})
import sys

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data



def ypr_to_quat(yaw, pitch, roll):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
    # Yaw then pitch then roll only!
    yaw = yaw / 180. * math.pi
    pitch = pitch / 180. * math.pi
    roll = roll / 180. * math.pi
    cy = math.cos(yaw * 0.5);
    sy = math.sin(yaw * 0.5);
    cp = math.cos(pitch * 0.5);
    sp = math.sin(pitch * 0.5);
    cr = math.cos(roll * 0.5);
    sr = math.sin(roll * 0.5);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;

    return [w, x, y, z]

def quat_to_ypr(quat):
    (w,x,y,z) = (quat[0], quat[1], quat[2], quat[3])
    sinr_cosp = 2 * (w * x + y * z);
    cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = math.atan2(sinr_cosp, cosr_cosp);

    sinp = 2 * (w * y - z * x);
    if abs(sinp)>= 1:
        pitch = math.pi / 2.0 * (1 if sinp >= 0 else -1)
    else:
        pitch = math.asin(sinp);

    siny_cosp = 2 * (w * z + x * y);
    cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = math.atan2(siny_cosp, cosy_cosp);

    return [yaw * 180 / math.pi, pitch * 180 / math.pi, roll* 180 / math.pi]

def qdist(q1, q2):
    return (sum(abs(q1[i] - q2[i]) for i in range(4)))


def data_to_quat(dataset):
    newdata = []
    for i in range(dataset.shape[0]):
        quat = ypr_to_quat(dataset[i, 0], dataset[i, 1], dataset[i, 2])
        otherquat = [val * -1 for val in quat]
        if (i > 0 and qdist(newdata[-1], quat) > qdist(newdata[-1], otherquat)):
            quat = otherquat
        newdata.append(quat)
    return np.array(newdata)


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx = np.random.permutation(min([len(ori_data), len(generated_data)]))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show(block=True)
        # plt.savefig("pca.pdf")

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show(block=True)
        # plt.savefig("tsne.pdf")


def load_longseq(filebase, seq_len):
    ori_datas = [np.loadtxt(filebase.format(idx), delimiter=",", skiprows=0) for idx in
                 range(1, 13)]
    # ori_datas = [data_to_quat(ori_data) for ori_data in ori_datas]
    ori_datas = [ori_data[::15] for ori_data in ori_datas]
    temp_data = []
    # Cut data by sequence length
    for ori_data in ori_datas:
        for i in range(0, len(ori_data) - seq_len, 1):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return np.array(data)

def load_orig(seq_len):
    return load_longseq('data/head{}.csv', seq_len)

def load_orig2():
    #seqlen 24
    quatdata =  np.array(pickle.load(open("oridataBASE.out", "rb")))
    print("minmax", np.min(quatdata, axis=(0,1)), np.max(quatdata, axis=(0,1)))
    # for i in range(quatdata.shape[0]):
    #     for t_idx in range(24):
    #         for ft_idx in [0, 3]:
    #             if quatdata[i, t_idx, ft_idx] > 0:
    #                 quatdata[i, t_idx, ft_idx] -= 1
    #             elif quatdata[i, t_idx, ft_idx] < 0:
    #                 quatdata[i, t_idx, ft_idx] += 1
        # plt.plot(quatdata[i,:,3])
        # plt.show(block=True)
        # plt.clf()
    # quatdata = (quatdata + 2)
    # quatdata = ((quatdata + 2) % 2) - 1
    return quatdata
    # TO YPR
    # newypr = []
    # for idx in range(quatdata.shape[0]):
    #     newypr.append([])
    #     for time_idx in range(24):
    #         newypr[-1].append(quat_to_ypr(quatdata[idx, time_idx]))
    # newdata = np.array(newypr)
    # print("Processed", newdata.shape)
    # return newdata

def load_fft(seq_len):
    data = load_longseq('fft/out{}.csv', seq_len)
    print("FFT",data.shape)
    return data


def load_timegan_long():
    [seqs_gen, _] = pickle.load(open("genseqs.out", "rb"))
    # TO YPR
    newypr = []
    for idx in range(len(seqs_gen)):
        newypr.append([])
        for time_idx in range(100):
            newypr[-1].append(quat_to_ypr(seqs_gen[idx][time_idx]))
    seqs_gen = np.array(newypr)
    return seqs_gen

def load_timegan_base():
    n_feat = 4 if do_quat else 3
    data = np.empty((0,24,n_feat))
    for i in range(0,1): #todo 8
        # with open("gensep{}.out".format(i),"rb") as dumpfile:
        with open("gendataX.out".format(i),"rb") as dumpfile:
            newdata = pickle.load(dumpfile)
            if do_quat:
                return newdata

            ### TEMP
            print("minmax", np.min(newdata, axis=(0,1)), np.max(newdata, axis=(0,1)))
            #newdata += np.array([180,90,180])
            #newdata %= np.array([360,180,360])
            #newdata -= np.array([180,90,180])
            return newdata

            print("minmax", np.min(newdata), np.max(newdata))

            # for i in range(newdata.shape[0]):
            #     for t_idx in range(24):
            #         for ft_idx in [0,3]:
            #             if newdata[i,t_idx,ft_idx] > 0:
            #                 newdata[i, t_idx, ft_idx] -= 1
            #             elif newdata[i,t_idx,ft_idx] < 0:
            #                 newdata[i, t_idx, ft_idx] += 1
            #             else:
            #                 print("zero")


            # print("ND TYPE", type(newdata), newdata.shape)
            # newdata = (newdata + 2)
            # print(np.min(newdata), np.max(newdata))
            # newdata = newdata % 2.0
            # print(np.min(newdata), np.max(newdata))
            # return newdata
            # newdata = ((newdata +2)%2)-1
            # newdata = np.roll(newdata, -1, axis=2)

            # #TO UNITY
            weights = np.sqrt(np.sum(np.square(newdata),axis=2))
            newdata /= np.repeat(weights[:,:,np.newaxis],4,2)

            # # TO YPR
            newypr = []
            for idx in range(newdata.shape[0]):
                newypr.append([])
                for time_idx in range(24):
                    newypr[-1].append(quat_to_ypr(newdata[idx, time_idx]))
            newdata = np.array(newypr)
            #with open("gendataONEYPR.out".format(i), "wb") as dumpfileYPR:
            #    pickle.dump(newdata, dumpfileYPR, protocol=4 )

            data = np.vstack((data, newdata))
    return data

def load_something(type):
    if type == "timegan_base":
        return load_timegan_base()
    elif type == "timegan_long":
        return load_timegan_long()
    elif type == "orig_base":
        return load_orig(24)
    elif type == "orig_long":
        return load_orig(100)
    elif type == "quatorig":
        return load_orig2()
    elif type == "fft_base":
        return load_fft(24)
    elif type == "fft_long":
        return load_fft(100)

def get_ls(type):
    if type.startswith("timegan_"):
        return "--"
    elif type.startswith("orig_"):
        return "-"
    elif type.startswith("fft_"):
        return ":"
    elif type.startswith("quatorig"):
        return "-"

colors = ["red", "green", "blue"]
def plot_hists(data, ls, minval, maxval, bkts, feature_idx=None):
    features = [feature_idx] if feature_idx is not None else range(data.shape[-1])
    data = np.sort(data,axis=0)
    print("sorted")
    # cdf = np.array(range(data.shape[0])) / data.shape[0]
    if (False and feature_idx is None and data.shape[-1] == 3): 
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.set_ylabel("Probability (yaw)")
        ax2.set_ylabel("Probability (pitch, roll)")
        ax.set_xlabel("Angle [°]")
    else:
        ax = plt.gca()
    for i in features:
        # pdf = np.gradient(cdf, data[:,i])
        # print(cdf[0:5])
        # print(data[0:5,i])
        # print(pdf[0:5])
        # goodvals = np.diff(data[:,i])
        # goodvals = goodvals.astype(bool)
        # print(goodvals.shape, np.sum(goodvals))
        #
        # goodvals = np.append(goodvals, True)
        # print(goodvals[430:440])
        # print(np.argwhere(goodvals[430:440]==False))
        # print(np.argwhere(goodvals==False))
        # goodvals = np.argwhere(goodvals)
        # for temp in range(1,len(data)):
        #     if data[temp,i] == data[temp-1,i]:
        #         print("jipla", temp)

        hist = np.histogram(data[:, i], bins=bkts, range=(minval, maxval), density=True)
        labels = (hist[1][1:] + hist[1][:-1]) / 2.0
        thisax = ax #if feature_idx is not None or i ==  0 else ax2
        thisax.plot(labels, hist[0] / 1, color=colors[i], ls=ls, zorder=-i)
        # print("SUM", sum(hist[0][x] / 1.0 for x in range(len(hist[0]))))
        # plt.plot(data[goodvals,i], pdf[goodvals], color=colors[i], ls=ls)



def plot_means(type, feature_idx=None):
    ls = get_ls(type)
    data = load_something(type)
    print("Data shape", type, data.shape)
    means = np.mean(data, axis=1)
    n_feat = 4 if do_quat else 3
    means = np.reshape(data, (-1,n_feat))
    print("reshaped")
    # means = np.reshape(data, (-1,4))
    # means = np.reshape(data, (-1,1))
    # #HACK for unity
    # means = np.sum(np.square(data), axis=2)
    # print(means[0])
    # means = np.expand_dims(means.flatten(), 1)
    # print(means[0:4])
    # print("HACK SH", means.shape)

    if do_quat:
        plot_hists(means, ls, -1, 1, 40, feature_idx)
    else:
        plot_hists(means, ls, -175, 175, 35, feature_idx)
    # plt.xlim([-1,1])
    print("Plotted")
def plot_ranges(type, feature_idx=None):
    ls = get_ls(type)
    data = load_something(type)
    ranges = np.max(data, axis=1) - np.min(data, axis=1)
    #print("ranges", data[0,:,0], ranges[0,0])
    plot_hists(ranges, ls, 0, 1 if do_quat else 360,90 , feature_idx)
def plot_autocor(type, feature_idx=None):
    ls = get_ls(type)
    data = load_something(type)
    autocors = [acf(np.abs(np.diff(data[i,:,feature_idx]))) for i in range(len(data))]
    autocors = np.mean(np.array(autocors), axis=0)
    plt.plot(np.arange(len(autocors))/15, autocors, ls=ls, color=colors[feature_idx], zorder=-feature_idx)
def plot_xcor(type, feature_idxs):
    xcolors={(0,1) : "goldenrod", (0,2): "magenta", (1,2): "teal"}
    ls = get_ls(type)
    data = load_something(type)
    (left, right) = feature_idxs
    xcors = [ccf(np.abs(np.diff(data[i,:,left])), np.abs(np.diff(data[i,:,right]))) for i in range(len(data))]
    #xcors = [scipy.signal.correlate(np.abs(np.diff(data[i,:,left])), np.abs(np.diff(data[i,:,right])), mode="valid") for i in range(len(data))]
    xcors = np.mean(np.array(xcors), axis=0)
    plt.plot(np.arange(10)/15,xcors[:10], ls=ls, color=xcolors[feature_idxs])
do_quat = False
length_type = "base"
features = [0,1,2,3,None] if do_quat else [None]
feature_names = ["yaw", "pitch", "roll"]
visualization(load_something("quatorig"), load_something("timegan_base"), 'tsne')
visualization(load_something("quatorig"), load_something("timegan_base"), 'pca')
# exit(1)
for feature in features:
    plot_means("timegan_" + length_type, feature)
    #plot_means("fft_" + length_type, feature)
    #if do_quat:
    plot_means("quatorig", feature)
    #else:
    #    plot_means("orig_" + length_type, feature)
    plt.xlabel("{} [°]".format(feature_names[feature] if feature is not None else "Angle"))
    plt.ylabel("Probability")

    #plt.plot([],[],color=colors[0], label="Yaw")
    #plt.plot([],[],color=colors[1], label="Pitch")
    #plt.plot([],[],color=colors[2], label="Roll")
    #plt.legend()
    if True and feature is None:
        plt.plot([],[],color="red",  label="Yaw")
        plt.plot([],[],color="green", label="Pitch")
        plt.plot([],[],color="blue", label="Roll")
        plt.legend()
    plt.tight_layout()
    #plt.savefig("plot_datadist_{}_fft.pdf".format(feature_names[feature]))
    #plt.savefig("OUT{}F.png".format(sys.argv[1]))
    #plt.show(block=True)
    plt.savefig("plot_angle_bonus.pdf")
    plt.clf()
    break
for feature in features:
    plot_ranges("timegan_" + length_type, feature)
    #plot_ranges("fft_" + length_type, feature)
    #if do_quat:
    plot_ranges("quatorig", feature)
    #else:
    #    plot_ranges("orig_" + length_type, feature)
    plt.xlabel("{} range per sample [°]".format(feature_names[feature] if feature is not None else "Angle"))
    plt.ylabel("Probability")
    if True and feature is None:
        plt.plot([],[],color="red",  label="Yaw")
        plt.plot([],[],color="green", label="Pitch")
        plt.plot([],[],color="blue", label="Roll")
        plt.legend()
    plt.tight_layout()
    #plt.savefig("plot_rangedist_{}_fft.pdf".format(feature_names[feature]))
    #plt.savefig("OUT{}R.png".format(sys.argv[1]))
    #exit(1)
    #plt.show(block=True)
    plt.savefig("plot_motion_bonus.pdf")
    plt.clf()

for feature in [0,1,2]:
    plot_autocor("timegan_"+length_type, feature)
    plot_autocor("quatorig", feature)
    #plot_autocor("fft_"+length_type, feature)
    plt.xlabel("Time lag [s]")
    plt.ylabel("Angular autocorrelation")#.format(feature_names[feature] if feature is not None else "Angular"))
    if True and feature == 2:
        plt.plot([],[],color="red",  label="Yaw")
        plt.plot([],[],color="green", label="Pitch")
        plt.plot([],[],color="blue", label="Roll")
        plt.legend()
    plt.tight_layout()
    #plt.savefig("plot_autocor_{}_fft.pdf".format(feature_names[feature]))
plt.savefig("plot_autocor_bonus.pdf")
#plt.show(block=True)
plt.clf()

for fts in [(0,1),(0,2),(1,2)]:
    plot_xcor("timegan_"+length_type, fts)
    plot_xcor("quatorig", fts)
    #plot_xcor("fft_"+length_type, fts)
    plt.xlabel("Time lag [s]")
    plt.ylabel("Angular cross-correlation")#.format(feature_names[fts[0]], feature_names[fts[1]]))
    if True and fts==(1,2):
        plt.plot([],[],color="goldenrod", label="Yaw-pitch")
        plt.plot([],[],color="magenta",  label="Yaw-roll")
        plt.plot([],[],color="teal", label="Pitch-roll")
        plt.legend()
    plt.tight_layout()
    #plt.savefig("plot_xcor_{}_{}_fft.pdf".format(feature_names[fts[0]], feature_names[fts[1]]))
plt.savefig("plot_xcor_bonus.pdf")
#plt.show(block=True)
plt.clf()

#visualization(data, seqs_gen, 'tsne')
#visualization(data, seqs_gen, 'pca')
