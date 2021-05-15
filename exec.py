import cv2
import numpy as np
from scipy.optimize import curve_fit

input_img = cv2.imread('./HSeq_londonbridge_1.ppm', 0).astype(np.float32)/255.

# number of comparable fine images
N = 3

# constrast threshold
extrema_threshold = 0.05

# points whose Cm is in the below range are regarded as edge (ignored)
lower_cm_threshold = 0.7
upper_cm_threshold = 1.5

# sigma of kernels
# (I just copied them from the thesis. I don't know how to calculate them.
#  You have to append values if increase N.)
filter_sigmas = [0.6, 1.209, 2.396, 4.788, 9.519, 18.9]

# compute sigma of fine images
dog_sigmas = []
for i in range(N+2):
    mu = filter_sigmas[i+1] / filter_sigmas[i]
    dog_sigmas.append(mu * filter_sigmas[i] * np.sqrt(2*np.log(mu)/(mu**2-1)))

# estimate a function to convert fine image ID into sigma
# (FFD regards fine image ID as the sigma coordinate when executes keypoint refinement by Taylor expansion.
#  After the refinement, it must be converted into the true sigma coordinate.)


def exp_curve(x, a, b):
    return a*np.exp(b*x)


exp_params, cov = curve_fit(exp_curve, np.arange(0, N+2), dog_sigmas)

# compute the initial kernel
gauss_kernel = cv2.getGaussianKernel(5, 0.6)
h0 = gauss_kernel*gauss_kernel.T

# compute upscaling kernels
h_filters = [h0]
for j in range(1, N+3):
    h = []
    h.append(1)
    for i in range(2**(j-1)-1):
        h.append(0)
    h.append(4)
    for i in range(2**(j-1)-1):
        h.append(0)
    h.append(6)
    for i in range(2**(j-1)-1):
        h.append(0)
    h.append(4)
    for i in range(2**(j-1)-1):
        h.append(0)
    h.append(1)

    h = np.array(h)/16
    h_filters.append(h.reshape([-1, 1]) * h.reshape([1, -1]))

# compute coarse images
coarse_imgs = []
prev_img = input_img
for j in range(N+3):
    prev_img = cv2.filter2D(prev_img, -1, h_filters[j])
    coarse_imgs.append(prev_img)

# compute fine images
fine_imgs = []
for j in range(N+2):
    img = coarse_imgs[j] - coarse_imgs[j+1]
    print('maximum response of fine image {}: {}'.format(j+1, np.max(img)))
    fine_imgs.append(img)

keypoints = []
for y in range(1, input_img.shape[0]-1):
    for x in range(1, input_img.shape[1]-1):
        for j in range(N):
            pre = fine_imgs[j]
            cur = fine_imgs[j+1]
            nxt = fine_imgs[j+2]

            if cur[y, x] < extrema_threshold-0.01:
                continue

            # keypoint refinement by Taylor expansion

            dDx = (cur[y, x+1] - cur[y, x-1]) / 2
            dDy = (cur[y+1, x] - cur[y-1, x]) / 2
            dDs = (nxt[y, x] - pre[y, x]) / 2
            dD = np.array([dDx, dDy, dDs])

            Hxx = cur[y, x+1] + cur[y, x-1] - 2*cur[y, x]
            Hyy = cur[y+1, x] + cur[y-1, x] - 2*cur[y, x]
            Hss = nxt[y, x] + pre[y, x] - 2*cur[y, x]

            Hxy = (cur[y+1, x+1] - cur[y+1, x-1] -
                   cur[y-1, x+1] + cur[y-1, x-1]) / 4
            Hxs = (nxt[y, x+1] - pre[y, x+1] -
                   nxt[y, x-1] + pre[y, x-1]) / 4
            Hys = (nxt[y+1, x] - pre[y+1, x] -
                   nxt[y-1, x] + pre[y-1, x]) / 4
            # if use 6-neighbors for cross derivatives
            # Hxy = (cur[y, x+1] + cur[y, x-1] + cur[y+1, x] +
            #        cur[y-1, x] - cur[y-1, x-1] - cur[y+1, x+1] - 2*cur[y, x])
            # Hxs = (cur[y, x+1] + cur[y, x-1] + nxt[y, x] +
            #        pre[y, x] - pre[y, x-1] - nxt[y, x+1] - 2*cur[y, x])
            # Hys = (nxt[y, x] + pre[y, x] + cur[y+1, x] +
            #        cur[y-1, x] - pre[y-1, x] - nxt[y+1, x] - 2*cur[y, x])

            H = np.array([
                [Hxx, Hxy, Hxs],
                [Hxy, Hyy, Hys],
                [Hxs, Hys, Hss]])

            dpos = -np.linalg.solve(H, dD)

            if np.abs(dpos[0]) < 0.5 and np.abs(dpos[1]) < 0.5 and np.abs(dpos[2]) < 0.5:
                response = cur[y, x] + dD@dpos/2
                pos = np.array([x, y, j+1]) + dpos
                cm = 1 - 4*(Hxx*Hyy - Hxy**2)/(Hxx + Hyy)**2
                if response > extrema_threshold and (cm <= lower_cm_threshold or upper_cm_threshold <= cm):
                    keypoints.append(
                        (response, pos[0], pos[1], exp_curve(pos[2], *exp_params)))


for row in sorted(keypoints, reverse=True):
    print(row)
