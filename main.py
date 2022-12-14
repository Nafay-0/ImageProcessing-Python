import cv2
import math
import numpy as np


# Huffman Coding
def HuffmanCoding(image):
    # get image size
    (h, w) = image.shape[:2]
    # create dictionary
    dictionary = {}
    # create histogram
    for i in range(0, h):
        for j in range(0, w):
            if image[i][j] in dictionary:
                dictionary[image[i][j]] += 1
            else:
                dictionary[image[i][j]] = 1
    # sort dictionary
    dictionary = sorted(dictionary.items(), key=lambda x: x[1])
    # create tree
    while len(dictionary) > 1:
        # get two smallest elements
        left = dictionary[0]
        right = dictionary[1]
        # create new node
        node = (left[0] + right[0], left[1] + right[1])
        # remove two smallest elements
        dictionary = dictionary[2:]
        # insert new node
        dictionary.append(node)
        # sort dictionary
        dictionary = sorted(dictionary, key=lambda x: x[1])
    # create code
    code = {}
    # create code for each element
    for i in range(0, h):
        for j in range(0, w):
            if image[i][j] not in code:
                code[image[i][j]] = ''
                # get code
                node = dictionary[0]
                while image[i][j] not in node[0]:
                    if image[i][j] in node[0][0]:
                        code[image[i][j]] += '0'
                        node = node[0][0]
                    else:
                        code[image[i][j]] += '1'
                        node = node[0][1]
    return code


# Run Length Encoding
def RunLengthEncoding(image):
    code = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if len(code) == 0:
                code.append([image[i][j], 1])
            else:
                if image[i][j] == code[-1][0]:
                    code[-1][1] += 1
                else:
                    code.append([image[i][j], 1])
    return code



#Kmeans image clustering
def kmeans_segment(img,k):
    # K = number of clusters
    # original image dimensions
    m = img.shape[0]
    n = img.shape[1]


    # converting image to 2D array
    img = img.reshape((-1, 3))
    # converting to np.float32
    img = np.float32(img)
    max_iterations = 15

    # random initialization of centroids
    centroids = np.random.randint(0, 255, size=(k, 3))
    # assign each pixel to a cluster
    clusters = np.zeros(img.shape[0], dtype=np.uint8)
    center_changed = True
    iteration = 0
    while center_changed and iteration < max_iterations:
        center_changed = False
        iteration += 1
        for i in range(img.shape[0]):
            distances = np.zeros(k)
            for j in range(k):
                distances[j] = np.linalg.norm(img[i] - centroids[j])
            cluster = np.argmin(distances)
            if clusters[i] != cluster:
                clusters[i] = cluster
                center_changed = True
        for i in range(k):
            centroids[i] = np.mean(img[clusters == i], axis=0)
    # assign each pixel to the cluster it belongs to
    segmented_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        segmented_img[i] = centroids[clusters[i]]
    # convert back to uint8
    segmented_img = np.uint8(segmented_img)
    # reshape back to the original image dimension
    segmented_img = segmented_img.reshape((m, n, 3))
    return segmented_img

# Connected Component Labeling
def ConnectedComponentLabeling(image):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # create label
    label = 1
    # create dictionary
    dictionary = {}
    # connected component labeling
    for i in range(0, h):
        for j in range(0, w):
            if image[i][j] != 0:
                # check left and up neighbor
                left = image[i][j - 1]
                up = image[i - 1][j]
                if left == 0 and up == 0:
                    # create new label
                    dictionary[label] = label
                    new_image[i][j] = label
                    label += 1
                elif left != 0 and up == 0:
                    # assign left label
                    new_image[i][j] = left
                elif left == 0 and up != 0:
                    # assign up label
                    new_image[i][j] = up
                else:
                    # assign min label
                    new_image[i][j] = min(left, up)
                    # update dictionary
                    dictionary[max(left, up)] = min(left, up)
    # update label
    for i in range(0, h):
        for j in range(0, w):
            if new_image[i][j] != 0:
                new_image[i][j] = dictionary[new_image[i][j]]
    return new_image

#HSV Segmentation
def HSVSegmentation(image):
    # convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    return res

# Region Growing Algorithm
def RegionGrowing(image,seed_count,threshold):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # create seed
    seed = np.zeros(image.shape)
    # create queue
    queue = []
    # create label
    label = 1
    # create dictionary
    dictionary = {}
    # create seed
    for i in range(0, seed_count):
        # get random seed
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        # add seed to queue
        queue.append((x, y))
        # add seed to dictionary
        dictionary[(x, y)] = label
        # add seed to seed
        seed[x][y] = 255
        # add label
        label += 1
    # region growing
    while len(queue) != 0:
        # get current pixel
        (x, y) = queue.pop(0)
        # get current label
        current_label = dictionary[(x, y)]
        # get current pixel value
        current_value = image[x][y]
        # check left neighbor
        if y - 1 >= 0:
            # get left neighbor value
            left_value = image[x][y - 1]
            # check threshold
            if abs(current_value - left_value) < threshold:
                # check left neighbor label
                if (x, y - 1) not in dictionary:
                    # add left neighbor to queue
                    queue.append((x, y - 1))
                    # add left neighbor to dictionary
                    dictionary[(x, y - 1)] = current_label
                    # add left neighbor to seed
                    seed[x][y - 1] = 255
        # check right neighbor
        if y + 1 < w:
            # get right neighbor value
            right_value = image[x][y + 1]
            # check threshold
            if abs(current_value - right_value) < threshold:
                # check right neighbor label
                if (x, y + 1) not in dictionary:
                    # add right neighbor to queue
                    queue.append((x, y + 1))
                    # add right neighbor to dictionary
                    dictionary[(x, y + 1)] = current_label
                    # add right neighbor to seed
                    seed[x][y + 1] = 255
        # check up neighbor
        if x - 1 >= 0:
            # get up neighbor value
            up_value = image[x - 1][y]
            # check threshold
            if abs(current_value - up_value) < threshold:
                # check up neighbor label
                if (x - 1, y) not in dictionary:
                    # add up neighbor to queue
                    queue.append((x - 1, y))
                    # add up neighbor to dictionary
                    dictionary[(x - 1, y)] = current_label
                    # add up neighbor to seed
                    seed[x - 1][y] = 255
        # check down neighbor
        if x + 1 < h:
            # get down neighbor value
            down_value = image[x + 1][y]
            # check threshold
            if abs(current_value - down_value) < threshold:
                # check down neighbor label
                if (x + 1, y) not in dictionary:
                    # add down neighbor to queue
                    queue.append((x + 1, y))
                    # add down neighbor to dictionary
                    dictionary[(x + 1, y)] = current_label
                    # add down neighbor to seed
                    seed[x + 1][y] = 255
    # update label
    for i in range(0, h):
        for j in range(0, w):
            if seed[i][j] != 0:
                new_image[i][j] = dictionary[(i, j)]
    return new_image



def split(img,Threshold):
    h = img.shape[0]
    w = img.shape[1]
    segments = []
    if np.max(img) - np.min(img) > Threshold:
        segments.append(img[0:h//2,0:w//2])
        segments.append(img[0:h//2,w//2:])
        segments.append(img[h//2:,0:w//2])
        segments.append(img[h//2:,w//2:])
    else:
        segments.append(img)
    return segments

def MergeSegments(segments,threshold):
    # merge segments based on similarity
    # if after merging the range of the merged segment is less than threshold, merge the segments
    # else return the segments
    if len(segments) == 1:
        return segments
    else:
        merged_segments = []
        for i in range(0,len(segments),2):
            if i+1 < len(segments):
                if np.max(segments[i]) - np.min(segments[i]) < threshold and np.max(segments[i+1]) - np.min(segments[i+1]) < threshold:
                    merged_segments.append(np.concatenate((segments[i],segments[i+1]),axis=1))
                else:
                    merged_segments.append(segments[i])
                    merged_segments.append(segments[i+1])
            else:
                merged_segments.append(segments[i])
        return MergeSegments(merged_segments,threshold)

# Guassian Filter
def GuassianFilter(image, kernel_size, sigma):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # create kernel
    kernel = np.zeros((kernel_size, kernel_size))
    # calculate kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = math.exp(-((i - kernel_size / 2) ** 2 + (j - kernel_size / 2) ** 2) / (2 * sigma ** 2))
    # normalization
    kernel = kernel / np.sum(kernel)
    # convolution
    for i in range(0, h):
        for j in range(0, w):
            for m in range(0, kernel_size):
                for n in range(0, kernel_size):
                    new_image[i][j] += image[i + m][j + n] * kernel[m][n]
    return new_image
def MedianFilter(image,kernel_size):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # median filter
    for i in range(0, h):
        for j in range(0, w):
            new_image[i][j] = np.median(image[i:i + kernel_size, j:j + kernel_size])
    return new_image

# Image Convolution
def convolution(image, kernel):
    # get image size
    (h, w) = image.shape[:2]
    # get kernel size
    (kh, kw) = kernel.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # convolution
    for i in range(0, h):
        for j in range(0, w):
            for m in range(0, kh):
                for n in range(0, kw):
                    new_image[i][j] += image[i + m][j + n] * kernel[m][n]
    return new_image

def Filtering(image,kernel):
    # get image size
    (h, w) = image.shape[:2]
    # get kernel size
    (kh, kw) = kernel.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # convolution
    for i in range(0, h):
        for j in range(0, w):
            for m in range(0, kh):
                for n in range(0, kw):
                    new_image[i][j] += image[i + m][j + n] * kernel[m][n]
    return new_image


# Otsu Thresholding
def OtsuThresholding(image):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # calculate probability
    prob = hist / (h * w)
    # calculate mean
    mean = np.zeros(256)
    for i in range(0, 256):
        mean[i] = np.sum([j * prob[j] for j in range(0, i + 1)])
    # calculate variance
    variance = np.zeros(256)
    for i in range(0, 256):
        variance[i] = np.sum([(j - mean[i]) ** 2 * prob[j] for j in range(0, i + 1)])
    # calculate threshold
    threshold = np.argmax(variance)
    # binarize image
    for i in range(0, h):
        for j in range(0, w):
            if image[i][j] > threshold:
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0
    return new_image

# Adaptive Thresholding
def AdaptiveThresholding(image, block_size, C):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # adaptive thresholding
    for i in range(0, h):
        for j in range(0, w):
            # calculate mean
            mean = np.mean(image[max(0, i - block_size // 2):min(h, i + block_size // 2),
                                 max(0, j - block_size // 2):min(w, j + block_size // 2)])
            # binarize image
            if image[i][j] > mean + C:
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0
    return new_image

# Mean Filter
def MeanFilter(image, kernel_size):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # mean filter
    for i in range(0, h):
        for j in range(0, w):
            new_image[i][j] = np.mean(image[i:i + kernel_size, j:j + kernel_size])
    return new_image

# Local Mean Thresholding
def LocalMeanThresholding(image, block_size):
    # get image size
    (h, w) = image.shape[:2]
    # create new image
    new_image = np.zeros(image.shape)
    # local mean thresholding
    for i in range(0, h):
        for j in range(0, w):
            # calculate mean
            mean = np.mean(image[max(0, i - block_size // 2):min(h, i + block_size // 2),
                                 max(0, j - block_size // 2):min(w, j + block_size // 2)])
            # binarize image
            if image[i][j] > mean:
                new_image[i][j] = 255
            else:
                new_image[i][j] = 0
    return new_image

#Histogram Equalization
def HistogramEqualization(image):
    # get image size
    (h, w) = image.shape[:2]
    # get histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # calculate cdf
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # create new image
    new_image = np.zeros(image.shape)
    # equalize image
    for i in range(0, h):
        for j in range(0, w):
            new_image[i][j] = cdf_normalized[image[i][j]]
    return new_image


# Histogram of Oriented Gradients
def HOG(image):
    # get image size
    (h, w) = image.shape[:2]
    # calculate gradient
    gx = cv2.Sobel(np.float32(image), cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(np.float32(image), cv2.CV_64F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # quantizing binvalues in (0...16)
    bins = np.int32(16 * ang / (2 * np.pi))
    # Divide to 4 sub-squares
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

# IOU calculation
def IOU(box1, box2):
    # box1: (x1, y1, x2, y2)
    # box2: (x1, y1, x2, y2)
    # calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = (x2 - x1) * (y2 - y1)
    # calculate union
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    # calculate IOU
    iou = intersection / union
    return iou


#Non-Maximum Suppression
def nms(boxes, scores, threshold):
    # boxes: (x1, y1, x2, y2)
    # scores: (score)
    # threshold: threshold of IOU
    # sort boxes by scores
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # calculate IOU
        iou = np.array([IOU(boxes[i], boxes[j]) for j in order[1:]])
        # delete boxes with IOU > threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    return keep

# Interpolation
def BilinearInterpolation(image,new_size):
    rows, cols = image.shape
    x_ratio = float(cols) / new_size[0]
    y_ratio = float(rows) / new_size[1]
    new_image = np.zeros(new_size)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            x = int(x_ratio * i)
            y = int(y_ratio * j)
            x_diff = (x_ratio * i) - x
            y_diff = (y_ratio * j) - y
            new_image[j][i] = (1 - x_diff) * (1 - y_diff) * image[y][x] + x_diff * (1 - y_diff) * image[y][x + 1] + (
                        1 - x_diff) * y_diff * image[y + 1][x] + x_diff * y_diff * image[y + 1][x + 1]
    return new_image


def NearestNeighbourInterpolation(image, new_size):
    # new_size: new size of image
    # get image size
    (h, w) = image.shape[:2]
    # get scaling factor
    row_ratio = w/new_size[0]
    col_ratio = h/new_size[1]
    # create new image
    new_image = np.zeros(new_size)
    # scale image
    for i in range(0, new_size[0]):
        for j in range(0, new_size[1]):
            # get coordinates of nearest neighbour
            x = math.floor(i*row_ratio)
            y = math.floor(j*col_ratio)
            # assign pixel value
            new_image[j][i] = image[y][x]
    return new_image




def detectCircle(image):
    # parametric equation of circle
    # x = a + r*cos(theta)
    # y = b + r*sin(theta)
    # radius range : 10 - 100
    threshold = 0.7
    for a in range(0, image.shape[0]):
        for b in range(0, image.shape[1]):
            for r in range(10, 100):
                count = 0
                for theta in range(0, 360):
                    x = a + r * math.cos(theta)
                    y = b + r * math.sin(theta)
                    if image[x][y] == 255:
                        count += 1
                if count / 360 > threshold:
                    print("Circle detected at: ", a, b, r)


def detectLine(image):
    # parametric equation of line
    # p = x*cos(theta) + y*sin(theta)
    # theta range : 0 - 360
    accumulator = np.zeros((image.shape[0]+image.shape[1], 360))
    for x in range(0, image.shape[1]):
        for y in range(0, image.shape[0]):
            if image[y][x] == 255:
                for theta in range(0, 360):
                    p = x * math.cos(theta) + y * math.sin(theta)
                    accumulator[int(p)][theta] += 1
    for theta in range(0, 360):
        for p in range(0, image.shape[0]+image.shape[1]):
            if accumulator[p][theta] > 100:
                print("Line detected at: ", p, theta)
                for x in range(0, image.shape[1]):
                    y = (p - x * math.cos(theta)) / math.sin(theta)
                    image[y][x] = 255
    return image


def ScaleImage(image, X, Y):
    # scale transformation matrix
    # [ 1 0 X ]
    # [ 0 1 Y ]
    # [ 0 0 1 ]
    # X: translation in x direction
    # Y: translation in y direction
    # scale image
    res_image = cv2.warpAffine(image, np.float32([[1, 0, X], [0, 1, Y], [0, 0, 1]]), (image.shape[0], image.shape[1]))
    return res_image


def rotateImage(image, angle):
    # rotate image
    # get image size
    (h, w) = image.shape[:2]
    # calculate center of image
    center = (w / 2, h / 2)
    # get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotate image
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def TrainLogisticRegression(X, Y):
    # X: input features as vector
    # Y: output labels as vector
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(X, Y)
    return clf


def HOGFeatures(image):
    # hog features
    hogDescriptor = cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    )
    features = hogDescriptor.compute(image)
    return features


def SIFTFeatures(image):
    # sift features
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors = sift.detectAndCompute(image, None)
    return keypoints1, descriptors


def SURFFeatures(image):
    # surf features
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors


def TrainSVM(X, Y):
    # X: input features as vector
    # Y: output labels as vector
    from sklearn import svm
    # grid search for best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'gamma': [0.1, 0.01], 'degree': [2, 3]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X, Y)
    # return best estimator -> best parameters
    return clf.best_estimator_


def TrainRandomForest(X, Y):
    # X: input features as vector
    # Y: output labels as vector
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X, Y)
    return clf


def TrainKNN(X, Y):
    # X: input features as vector
    # Y: output labels as vector
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, Y)
    return neigh


if __name__ == '__main__':
    image = cv2.imread("image.png", 0)
    detectCircle(image)
    detectLine(image)
    HOGFeatures(image)
    SIFTFeatures(image)
    SURFFeatures(image)
