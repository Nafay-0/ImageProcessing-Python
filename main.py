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
                    x = a + r*math.cos(theta)
                    y = b + r*math.sin(theta)
                    if image[x][y] == 255:
                        count += 1
                if count/360 > threshold:
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
                y = m*x + c
                if image[x][y] == 255:
                    count += 1
            if count/image.shape[0] > threshold:
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



if __name__ == '__main__':
    image = cv2.imread("image.png", 0)
    detectCircle(image)
    detectLine(image)
    HOGFeatures(image)
    SIFTFeatures(image)
    SURFFeatures(image)

