import cv2
import math
import numpy as np



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
    # y = mx + c
    # slope range : -100 -> 100
    # intercept range : 0 -> 100
    threshold = 0.7
    for m in range(-100, 100):
        for c in range(0, 100):
            count = 0
            for x in range(0, image.shape[0]):
                y = m * x + c
                if image[x][y] == 255:
                    count += 1
            if count / image.shape[0] > threshold:
                print("Line detected at: ", m, c)


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
