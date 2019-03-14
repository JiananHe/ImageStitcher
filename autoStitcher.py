import cv2
import numpy
import matplotlib.pyplot as plt

assert cv2.__version__.split('.')[0] == '3'
sift = cv2.xfeatures2d.SIFT_create()


def draw_matches(img1, img2, pts1, pts2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1 + w2, 3), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    p1 = numpy.int32([kpp for kpp in pts1])
    p2 = numpy.int32([kpp for kpp in pts2])

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.circle(vis, (x1, y1), 3, green, -1)
        cv2.circle(vis, (w1 + x2, y2), 3, green, -1)

        cv2.line(vis, (x1, y1), (w1 + x2, y2), red, 2)

    cv2.imshow('matches points pair', vis)
    cv2.waitKey(0)


def combine_images(img1, img2, h_matrix):
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))
    
    points2 = numpy.array(
        [[0, 0], [0, img2.shape[0]], [img2.shape[1], img2.shape[0]], [img2.shape[1], 0]], dtype=numpy.float32)
    points2 = points2.reshape((-1, 1, 2))
    
    points1 = cv2.perspectiveTransform(points1, h_matrix)
    points = numpy.concatenate((points1, points2), axis=0)
    [x_min, y_min] = numpy.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    cv2.imshow('warped image', cv2.resize(output_img, (640, 360)))

    output_img[-y_min:img1.shape[0] - y_min, -x_min:img1.shape[1] - x_min] = img1
    cv2.imshow('stitched image', cv2.resize(output_img, (640, 360)))
    cv2.waitKey(0)

    return output_img


if __name__ == "__main__":
    # read source picture
    src1 = cv2.imread(r'D:\Code\GasTank\examples\example12\p1.jpg')
    src1 = cv2.resize(src1, (1280, 720))
    cv2.imshow('src1', src1)
    img1 = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)

    src2 = cv2.imread(r'D:\Code\GasTank\examples\example12\p2.jpg')
    src2 = cv2.resize(src2, (1280, 720))
    cv2.imshow('src2', src2)
    img2 = cv2.cvtColor(src2, cv2.COLOR_RGB2GRAY)

    # extract key points
    features1 = sift.detectAndCompute(img1, None)
    features2 = sift.detectAndCompute(img2, None)

    # key points match with flann algorithm
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    keypoints1, descriptors1 = features1
    keypoints2, descriptors2 = features2
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    print("matches: " + str(matches))

    # matched points refining
    lowe = 0.7
    positive = []
    for match1, match2 in matches:
        if match1.distance < lowe * match2.distance:
            positive.append(match1)

    pts1 = numpy.array([keypoints1[good_match.queryIdx].pt for good_match in positive], dtype=numpy.float32)
    src_pts = pts1.reshape((-1, 1, 2))
    pts2 = numpy.array([keypoints2[good_match.trainIdx].pt for good_match in positive], dtype=numpy.float32)
    dst_pts = pts2.reshape((-1, 1, 2))

    draw_matches(src1, src2, pts1, pts2)

    assert len(positive) > 0
    # calculate transformation matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(H)

    # combine
    output_img = combine_images(src1, src2, H)

    # save or display
    cv2.imwrite(r'D:\Code\GasTank\demo.jpg', output_img)