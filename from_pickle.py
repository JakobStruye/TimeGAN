import pickle
from metrics.visualization_metrics import visualization
from timegan import generate_data
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def distance(left, right):
    #return np.max(np.abs(left[-5:,:] - right[:5,:]))
    return np.mean(np.square(left[-5:,:] - right[:5,:]))

data = []
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    [no, z_dim, ori_time, max_seq_len, ori_data, max_val, min_val, Z_mb] = pickle.load(open("vars.pkl", "rb"))

    graph = tf.get_default_graph()
    Z = graph.get_tensor_by_name("myinput_z:0")
    X = graph.get_tensor_by_name("myinput_x:0")
    T = graph.get_tensor_by_name("myinput_t:0")
    X_hat = graph.get_tensor_by_name("X_hat:0")

    for i in range(750):
        generated_data = generate_data(sess, no, z_dim, ori_time, max_seq_len, X_hat, Z, X, T, Z_mb, ori_data, max_val, min_val)
        data.extend(generated_data)
        print(len(data))
    # visualization(ori_data, generated_data, 'pca')
    # visualization(ori_data, generated_data, 'tsne')

random.shuffle(data)
seqs = []
fulls = []
for i in range(1000):
    cur = data.pop(0)
    full = [cur]
    for j in range(4):
        mindist = 999
        for (idx,candidate) in enumerate(data):
            dist = distance(cur, candidate)
            if dist < mindist:
                mindist = dist
                winner_idx = idx
        print(mindist, winner_idx)
        if (mindist > 0.001):
            break
        next_entry = data.pop(winner_idx)
        cur[-5,:] = 0.9 * cur[-5,:] + 0.1 * next_entry[0]
        cur[-4, :] = 0.7 * cur[-4, :] + 0.3 * next_entry[1]
        cur[-3, :] = 0.5 * cur[-3, :] + 0.5 * next_entry[2]
        cur[-2, :] = 0.3 * cur[-2, :] + 0.7 * next_entry[3]
        cur[-1, :] = 0.1 * cur[-1, :] + 0.9 * next_entry[4]
        cur = np.vstack([cur, next_entry[5:,:]])
        full.append(next_entry)
        plt.axvline(24 + (j*19), color="grey")
    if len(full) < 5:
        continue
    seqs.append(cur)
    fulls.append(full)
    plt.plot(cur)
    plt.savefig("gen{}.pdf".format(i))
    plt.clf()
pickle.dump([seqs, fulls], open("genseqs.out", "wb"))
