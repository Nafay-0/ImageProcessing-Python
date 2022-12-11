import cv2
import math


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
