import pickle
from metrics.visualization_metrics import visualization
from timegan import generate_data
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import ctypes
import math

np.set_printoptions(suppress=True)

def distance(left, right):
    #return np.max(np.abs(left[-5:,:] - right[:5,:]))
    return np.mean(np.square(left[-5:,:] - right[:5,:]))

def tonumpyarray(mp_arr, dtype=float):
    return np.frombuffer(mp_arr.get_obj(), dtype=dtype)

do_combine = False
gen_loops = 1
samples_per = 14112
samples = gen_loops * samples_per
datashape = (samples, 24, 1) #TODO back to 4

if do_combine:
    shared_arr = mp.Array(ctypes.c_double, datashape[0] * datashape[1] * datashape[2])
    data_arr = tonumpyarray(shared_arr)
    data_arr = data_arr.reshape(datashape)
else:
    data_arr = np.zeros(datashape)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    [no, z_dim, ori_time, max_seq_len, ori_data, max_val, min_val, Z_mb] = pickle.load(open("vars.pkl", "rb"))

    graph = tf.get_default_graph()
    Z = graph.get_tensor_by_name("myinput_z:0")
    X = graph.get_tensor_by_name("myinput_x:0")
    T = graph.get_tensor_by_name("myinput_t:0")
    X_hat = graph.get_tensor_by_name("X_hat:0")

    for i in range(gen_loops):
        print(i)
        generated_data = generate_data(sess, no, z_dim, ori_time, max_seq_len, X_hat, Z, X, T, Z_mb, ori_data, max_val, min_val)
        print("DATA",generated_data[0,:,:])
        data_arr[i*samples_per: (i+1) * samples_per] = generated_data
    # visualization(ori_data, generated_data, 'pca')
    # visualization(ori_data, generated_data, 'tsne')

print(data_arr[-1])
random.shuffle(data_arr)
n_slices = math.ceil(gen_loops / 100)
per_slice = math.ceil(data_arr.shape[0] / n_slices)
print("{} slices of {} each for {} total".format(n_slices, per_slice, data_arr.shape))
for i in range(n_slices):
    slice_arr = data_arr[i*per_slice:(i+1)*per_slice]
    with open("gensep{}.out".format(i), "wb") as dumpfile:
        pickle.dump(slice_arr,dumpfile, protocol=4)
if not do_combine:
    exit(0)
shared_used = mp.Array("b", samples)
used_arr = tonumpyarray(shared_used, bool)
used_arr[:] = 0

def combine_seqs(count):
    data = tonumpyarray(shared_arr)
    data = data.reshape(datashape)
    used = tonumpyarray(shared_used, bool)
    print(data.shape, used.shape)
    seqs = []
    fulls = []
    for ctr in range(count):
        print(ctr)
        while True:
            start_idx = random.randrange(data.shape[0])
            if not used[start_idx]:
                break

        cur = data[start_idx]
        used[start_idx] = True
        full = [cur]
        for j in range(4):
            mindist = 999
            for idx in range(data.shape[0]):
                if not used[idx]:
                    candidate = data[idx]
                    dist = distance(cur, candidate)
                    if dist < mindist:
                        mindist = dist
                        winner_idx = idx
            if (mindist > 0.001):
                print("skip", j)
                break
            used[winner_idx] = True
            next_entry = data[winner_idx]
            cur[-5,:] = 0.9 * cur[-5,:] + 0.1 * next_entry[0]
            cur[-4, :] = 0.7 * cur[-4, :] + 0.3 * next_entry[1]
            cur[-3, :] = 0.5 * cur[-3, :] + 0.5 * next_entry[2]
            cur[-2, :] = 0.3 * cur[-2, :] + 0.7 * next_entry[3]
            cur[-1, :] = 0.1 * cur[-1, :] + 0.9 * next_entry[4]
            cur = np.vstack([cur, next_entry[5:,:]])
            full.append(next_entry)
            # plt.axvline(24 + (j*19), color="grey")
        if len(full) < 5:
            continue
        seqs.append(cur)
        fulls.append(full)
        # plt.plot(cur)
        # plt.savefig("gen{}.pdf".format(i))
        # plt.clf()
    print(np.sum(used))
    return (seqs, fulls)

# threads = []
# threads.append(threading.Thread(target=combine_seqs, args=(0,0,1)))
# threads.append(threading.Thread(target=combine_seqs, args=(1,1,2)))
# threads.append(threading.Thread(target=combine_seqs, args=(2,2,3)))
# threads.append(threading.Thread(target=combine_seqs, args=(3,3,4)))
# for t in threads:
#     t.start()
# for t in threads:
#     t.join()

poolsize = 16
sample_count = 1024
with mp.Pool(poolsize) as p:
    args = []
    outputs = p.map(combine_seqs, poolsize*[sample_count//poolsize])


seqs = [seq for output in outputs for seq in output[0]] #flatten
fulls = [full for output in outputs for full in output[1]] #flatten

pickle.dump([seqs, fulls], open("genseqs.out", "wb"))
