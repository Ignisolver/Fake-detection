import cv2 as cv
from PIL.Image import ImageTransformHandler
from PIL import Image
from scipy.ndimage import sobel
import numpy as np


def canny(img, data=None):
    # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    img = np.array(img)
    filtrated = cv.Canny(img, 100, 200)
    return Image.fromarray(filtrated)


def my_sobel(img, data=None):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html
    img = np.array(img)
    filtrated = sobel(img)
    return Image.fromarray(filtrated)


def harris(img, data=None):
    # https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    img = np.array(img)
    if data == None:
        data = (2, 3, 0.04)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, *data)
    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    return Image.fromarray(dst)


def shi_tomas(img, data=None):
    #https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
    img = np.array(img)
    if data == None:
        data = (40, 0.1, 10)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, *data)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
    return Image.fromarray(img)


def fast(img, data=None):
    #https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html
    img = np.array(img)
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    img_2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    return Image.fromarray(img_2)


def laplacian_of_gaussian(img, data=None):
    # https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
    img = np.array(img)
    ddepth = cv.CV_16S
    kernel_size = 3
    src = cv.GaussianBlur(img, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    return Image.fromarray(dst)


def ridge(img, data=None):
    img = np.array(img)
    image = img
    ridge_filter = cv.ximgproc.RidgeDetectionFilter_create()
    ridges = ridge_filter.getRidgeFilteredImage(image)
    print(type(ridges), ridges.shape, ridges[0:10, 0:10])
    return Image.fromarray(ridges)


def fourier(img, data=None):
    img = np.array(img)
    src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    transformed = np.fft.fftshift(np.fft.fft2(src_gray))
    t = np.log(abs(transformed))
    t = t.astype(np.int8) * 10
    return Image.fromarray(t)
