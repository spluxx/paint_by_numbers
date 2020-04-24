import numpy as np
import numpy.linalg as la
from skimage import filters, color, measure, morphology, transform, io
from sklearn import cluster
from scipy import ndimage
import time

# constants
max_patch_size = 20*20
palette_sigma_max = 20
sigma_max = 5
contour_scale = 3
chunk_size = 200


def log_time_delta(t0, t1):
    print("Finished in {:0.2f}s".format(t1-t0))


def map_pixels(img, palette, C):
    # map each pixel in I to the closest (in LAB space) color in the palette
    H, W, _ = img.shape
    C = palette.shape[0]

    img_s = img.reshape(H, W, 1, 3)
    palette_s = palette.reshape(1, 1, C, 3)
    diff = img_s - palette_s  # (H, W, C, 3)
    dist = la.norm(diff, axis=3)  # (H, W, C)
    indices = np.argmin(dist, axis=2)  # (H, W)

    return palette.reshape(C, 3)[indices]


def generate_palette(img, C, s):
    I_blurred = filters.gaussian(
        img, sigma=palette_sigma_max*s, multichannel=True)

    # Run clustering in LAB space
    sample_ratio = 0.01
    sample_size = np.floor(I_blurred.size / 3 * sample_ratio).astype('int')

    # convert to LAB space
    I_lab = color.rgb2lab(I_blurred, illuminant='D65')
    I_lab_vec = np.reshape(I_lab, (np.floor(I_lab.size/3).astype(int), 3))

    # sample uniformly from I_lab_vec
    samples = I_lab_vec[np.random.choice(I_lab_vec.shape[0], sample_size)]

    # clustering
    res = cluster.AgglomerativeClustering(n_clusters=C).fit(samples)

    # Get the center LAB color for each cluster
    mask = res.labels_.reshape(sample_size, 1) == np.arange(C).reshape(1, C)
    return np.array([np.mean(samples[mask[:, c]], axis=0) for c in range(C)])


# identify and merge tiny patches
def merge_tiny(img, th, palette, C):
    H, W, _ = img.shape

    # To use measure.label, we need to convert the image to single-channel
    # labeled by palette index
    palette_index = np.argmax(
        np.all(img.reshape(H, W, 1, 3) == palette.reshape(1, 1, C, 3), axis=3), axis=2)
    # labeled by unique number for each patch
    labeled, n_label = measure.label(
        palette_index, return_num=True, background=-1, connectivity=1)
    label_to_palette_idx = {
        label: p_idx
        for [label, p_idx]
        in np.unique(np.stack([labeled, palette_index], axis=2).reshape(labeled.size, 2), axis=0)
    }

    # gather labels that have small patch sizes
    labels, size = np.unique(labeled.reshape(labeled.size), return_counts=True)
    small_patch_labels = labels[np.nonzero(size < th)[0]]

    struct = ndimage.generate_binary_structure(2, 1)

    def merge(labeled_img, to_merge):
        mask = labeled_img == to_merge
        y, x = np.where(mask)
        # We only operate on the relevant part of the mask to speed things up,
        # since the mask is usually compact and small.
        ymin = np.max([np.min(y)-5, 0])
        ymax = np.min([np.max(y)+5, H-1])
        xmin = np.max([np.min(x)-5, 0])
        xmax = np.min([np.max(x)+5, W-1])
        mini_mask = mask[ymin:ymax+1, xmin:xmax+1]
        mini_img = labeled_img[ymin:ymax+1, xmin:xmax+1]
        # XOR mask with its dilation to create an exterior boundary
        edge = mini_mask ^ ndimage.binary_dilation(mini_mask, struct)

        # merge to closest color out of the neighbors
        color = palette[label_to_palette_idx[to_merge]]
        mini_oimg = img[ymin:ymax+1, xmin:xmax+1]
        edge_y, edge_x = np.nonzero(edge)
        edge_colors = mini_oimg[edge_y, edge_x]
        closest_idx = np.argmin(la.norm(edge_colors - color, axis=1))
        y = edge_y[closest_idx]
        x = edge_x[closest_idx]
        p = mini_img[y][x]

        new_mini_img = mini_mask * p + (1-mini_mask) * mini_img
        labeled_img[ymin:ymax+1, xmin:xmax+1] = new_mini_img

        return labeled_img

    merged = labeled.copy()
    for spl in small_patch_labels:
        merged = merge(merged, spl)

    merged_f = merged.flatten()
    merged_p = np.zeros(merged.size).astype('int')
    for i in range(merged.size):
        merged_p[i] = label_to_palette_idx[merged_f[i]]
    merged_p = merged_p.reshape(merged.shape)

    return palette[merged_p]


def generate_contour(img, scale, palette, C):
    def make_contour(img, scale):
        H, W, _ = img.shape

        rH = scale * H
        rW = scale * W

        # order 0 to preserve topology
        scaled_img = map_pixels(transform.resize(
            img, (rH, rW, 3), order=0), palette, C)

        palette_index = np.argmax(np.all(scaled_img.reshape(
            rH, rW, 1, 3) == palette.reshape(1, 1, C, 3), axis=3), axis=2)
        # labeled by unique number for each patch
        labeled, n_label = measure.label(palette_index, return_num=True)

        struct = ndimage.generate_binary_structure(2, 2)

        res = np.zeros((rH, rW)).astype('bool')
        for label in range(n_label):
            mask = labeled == label
            y, x = np.where(mask)
            if np.sum(mask.flatten()) == 0:
                continue
            # We only operate on the relevant part of the mask to speed things up,
            # since the mask is usually compact and small.
            ymin = np.max([np.min(y)-5, 0])
            ymax = np.min([np.max(y)+5, rH-1])
            xmin = np.max([np.min(x)-5, 0])
            xmax = np.min([np.max(x)+5, rW-1])
            mini_mask = mask[ymin:ymax+1, xmin:xmax+1]
            # XOR mask with its erosion to create an inner boundary
            edge = mini_mask ^ ndimage.binary_erosion(mini_mask, struct)
            res[ymin:ymax+1, xmin:xmax+1] |= edge

        skeleton = morphology.skeletonize(res)

        skeleton = np.pad(skeleton[1:rH-1, 1:rW-1], ((1, 1), (1, 1)), 'edge')

        return skeleton

    H, W, _ = img.shape

    rH = scale * H
    rW = scale * W

    final_img = np.zeros((rH, rW))

    y = np.append(np.arange(0, H, chunk_size), H)
    x = np.append(np.arange(0, W, chunk_size), W)

    for i in range(len(y)-1):
        for j in range(len(x)-1):
            min_y = np.max([y[i]-5, 0])
            max_y = np.min([y[i+1]+5, H])
            min_x = np.max([x[j]-5, 0])
            max_x = np.min([x[j+1]+5, W])

            final_img[
                scale*min_y:scale*max_y,
                scale*min_x:scale*max_x
            ] = make_contour(img[min_y:max_y, min_x:max_x], scale)

    return final_img


def pbn(I, C, s):
    t0 = time.time()

    print("Generating palette...")
    P_lab = generate_palette(I, C, s)

    t1 = time.time()
    log_time_delta(t0, t1)

    I_blurred = filters.gaussian(I, sigma=sigma_max*s, multichannel=True)
    I_lab = color.rgb2lab(I_blurred, illuminant='D65')
    I_P_lab = map_pixels(I_lab, P_lab, C)

    result = I_P_lab.copy()
    threshold = s * max_patch_size

    print("Merging tiny patches...")
    for th in np.linspace(3, threshold, 5):
        result = merge_tiny(result, th, P_lab, C)

    t2 = time.time()
    log_time_delta(t1, t2)

    print("Generating contour...")
    contour = generate_contour(result, contour_scale, P_lab, C)

    t3 = time.time()
    log_time_delta(t2, t3)

    return result, contour


def export_result(dir, name, colored, contour):
    rgb = 255*color.lab2rgb(colored, illuminant='D65')

    io.imsave("{}/{}.jpg".format(dir, name), rgb)
    io.imsave("{}/{}_contour.jpg".format(dir, name), 255*(1-contour))
