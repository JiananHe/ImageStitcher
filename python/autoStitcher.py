import cv2
import numpy
import os
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
    p2 = points2

    warped_corner = {}

    points2 = cv2.perspectiveTransform(points2, h_matrix)
    points = numpy.concatenate((points1, points2), axis=0)
    [x_min, y_min] = numpy.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    warped_img = cv2.warpPerspective(img2, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))

    p2 = cv2.perspectiveTransform(p2, H_translation.dot(h_matrix)).reshape(-1, 2)
    # cv2.circle(warped_img, (p2[0][0], p2[0][1]), 10, (0, 255, 0), -1)
    # cv2.circle(warped_img, (p2[1][0], p2[1][1]), 10, (0, 255, 0), -1)
    # cv2.circle(warped_img, (p2[2][0], p2[2][1]), 10, (0, 255, 0), -1)
    # cv2.circle(warped_img, (p2[3][0], p2[3][1]), 10, (0, 255, 0), -1)
    p2 = numpy.array([[min(p2[0][0], p2[1][0]), max(p2[2][0], p2[3][0])],
                     [min(p2[0][1], p2[1][1]), max(p2[2][1], p2[3][1])]])
    cv2.imshow('warped image', cv2.resize(warped_img, (int(warped_img.shape[1] * 360 / warped_img.shape[0]), 360)))

    stitched_img = warped_img.copy()

    stitched_img[-y_min:img1.shape[0] - y_min, -x_min:img1.shape[1] - x_min] = img1
    p1 = numpy.array([[-x_min, img1.shape[1] - x_min], [-y_min, img1.shape[0] - y_min]])
    cv2.imshow('stitched image', cv2.resize(stitched_img, (int(stitched_img.shape[1] * 360 / stitched_img.shape[0]), 360)))

    output_img = optimize_combined_images(img1, warped_img, p1, p2, stitched_img)

    cv2.imshow('optimized image', cv2.resize(output_img, (int(output_img.shape[1] * 360 / output_img.shape[0]), 360)))
    cv2.waitKey(0)

    return stitched_img


def optimize_combined_images(img1, warped_img, p1, p2, stitched_img):  # p: [[x_min, x_max], [y_min, y_max]]
    # overlap
    start_x = int(p2[0][0] + 0.5)  # x_min of p2
    end_x = int(p1[0][1] + 0.5)    # x_max of p1
    start_y = int(min(p2[1][0], p1[1][0]) + 0.5)  # min of p1 y_min and p2 y_min
    end_y = int(max(p2[1][1], p1[1][1]) + 0.5)  # max of p1 y_max and p2 y_max
    overlap_width = end_x - start_x

    alpha_matrix = numpy.zeros((end_y - start_y, overlap_width, 3))
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            if (warped_img[i, j, :] == 0).all():
                alpha_matrix[i - start_y, j - start_x, :] = 1
            else:
                alpha_matrix[i - start_y, j - start_x, :] = 1 - (j - start_x) / overlap_width

    at = stitched_img[start_y:end_y, start_x:end_x, :] * alpha_matrix
    bt = warped_img[start_y:end_y, start_x:end_x, :] * (1 - alpha_matrix)
    stitched_img[start_y:end_y, start_x:end_x, :] = at + bt

    return stitched_img


if __name__ == "__main__":
    # read source picture
    src_dir = r'D:\Code\GasTank\examples\example6'
    src_index_begin = 1  # the name of the file must be incremented from left to right
    src_index_end = 2

    src1_name = str(src_index_begin) + ".jpg"
    src1_path = os.path.join(src_dir, src1_name)
    src1 = cv2.imread(src1_path)

    h1, w1 = src1.shape[:2]
    print("read source image(width=%d, height=%d)" % (w1, h1))
    ws1 = int(w1 * 720 / h1)
    src1 = cv2.resize(src1, (ws1, 720))  # scale
    
    for idx in range(src_index_begin + 1, src_index_end + 1):
        img1 = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)

        src2_name = str(idx) + ".jpg"
        src2_path = os.path.join(src_dir, src2_name)
        src2 = cv2.imread(src2_path)

        h2, w2 = src2.shape[:2]
        print("read source image(width=%d, height=%d)" % (w2, h2))
        ws2 = int(w2 * 720 / h2)
        src2 = cv2.resize(src2, (ws2, 720))
        img2 = cv2.cvtColor(src2, cv2.COLOR_RGB2GRAY)
        
        print("****** begin stitch......\n")
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

        # draw_matches(src1, src2, pts1, pts2)

        assert len(positive) > 0
        # calculate transformation matrix
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        print(H)
        # save
        numpy.savetxt(os.path.join(src_dir, str(idx) + "_h_matrix_auto.txt"), H)

        # combine
        result_img = combine_images(src1, src2, H)
        src1 = result_img

    # save or display
    cv2.imwrite(r'D:\Code\GasTank\demo.jpg', result_img)