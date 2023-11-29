import pickle
import matplotlib.pyplot as plt
import sys
import math

def quat_to_ypr(w,x,y,z):
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
[seqs, fulls] = pickle.load(open("genseqs.out", "rb"))
do_ypr = False if len(sys.argv) == 1 else sys.argv[1] == "ypr"
for x in range(len(seqs)):
    seq = seqs[x]
    full = fulls[x]
    if do_ypr:
        seq = [quat_to_ypr(*quat) for quat in seq]
    plt.plot(seq)
    if not do_ypr:
        for i in range(len(full)):
            plt.gca().set_prop_cycle(None)
            plt.plot(range(i*19, i*19+24), full[i], lw=4, alpha=0.5)
    plt.savefig("gen{}{}.png".format(x, "_ypr" if do_ypr else ""))
    plt.clf()
