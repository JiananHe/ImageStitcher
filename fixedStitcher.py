import cv2
import numpy
import os

# flag show the next click
click_flag = -1  # -1 for left, 1 for right
rule_flag = -1  # -1 for left, 1 for right


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    left_key_points = param["left_key_points"]
    right_key_points = param["right_key_points"]
    vis = param["vis"]
    w1 = param["w1"]

    global click_flag, rule_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > w1:  # right
            click_flag = -1
            right_key_points.append([x - w1, y])
        else:  # left
            click_flag = 1
            left_key_points.append([x, y])

        rule_flag = -rule_flag
        if click_flag == rule_flag:
            print("flag=%d, x=%d, y=%d" % (-rule_flag, x if x <= w1 else x - w1, y))
            cv2.circle(vis, (x, y), 3, (0, 255, 0), thickness=-1)
        else:
            raise RuntimeError('click error!!')
            cv2.destroyAllWindows()


def draw_keyPoints_pairs(src1, src2, idx):
    left_key_points = []
    right_key_points = []
    h1, w1 = src1.shape[:2]
    h2, w2 = src2.shape[:2]

    vis = numpy.zeros((max(h1, h2), w1 + w2, 3), numpy.uint8)
    vis[:h1, :w1] = src1
    vis[:h2, w1:w1 + w2] = src2

    # callback function, get the coordinates where mouse click
    cv2.namedWindow('image')
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param={"vis": vis, "left_key_points": left_key_points,
                                                               "right_key_points": right_key_points, "w1": w1})
    while (True):
        try:
            cv2.imshow("image", vis)
            if cv2.waitKey(100) & 0xFF == 27:  # esc exit
                break
        except Exception:
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

    print(left_key_points)
    print(right_key_points)
    assert len(left_key_points) == len(right_key_points) and len(left_key_points) >= 4

    left_key_points = numpy.array(left_key_points).reshape((-1, 1, 2))
    right_key_points = numpy.array(right_key_points).reshape((-1, 1, 2))

    # calculate homography matrix according to left-right key points pair
    H, mask = cv2.findHomography(right_key_points, left_key_points, cv2.RANSAC, 5.0)
    print(H)
    # save homography matrix
    numpy.savetxt(str(idx) + "_h_matrix.txt", H)
    return H


def combine_images(img1, img2, h_matrix):
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))

    points2 = numpy.array(
        [[0, 0], [0, img2.shape[0]], [img2.shape[1], img2.shape[0]], [img2.shape[1], 0]], dtype=numpy.float32)
    points2 = points2.reshape((-1, 1, 2))

    points2 = cv2.perspectiveTransform(points2, h_matrix)
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
    src_dir = r'D:\Code\GasTank\examples\example12'
    src_index_begin = 1  # the name of the file must be incremented from left to right
    src_index_end = 3

    src1_name = str(src_index_begin) + ".jpg"
    src1_path = os.path.join(src_dir, src1_name)
    src1 = cv2.imread(src1_path)

    h1, w1 = src1.shape[:2]
    print("read source image(width=%d, height=%d)" % (w1, h1))
    ws1 = int(w1*720/h1)
    src1 = cv2.resize(src1, (ws1, 720))  # scale
    img1 = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)  # gray
    for idx in range(src_index_begin + 1, src_index_end + 1):
        src2_name = str(idx) + ".jpg"
        src2_path = os.path.join(src_dir, src2_name)
        src2 = cv2.imread(src2_path)

        h2, w2 = src2.shape[:2]
        print("read source image(width=%d, height=%d)" % (w2, h2))
        ws2 = int(w2 * 720 / h2)
        src2 = cv2.resize(src2, (ws2, 720))
        img2 = cv2.cvtColor(src2, cv2.COLOR_RGB2GRAY)

        print("****** begin stitch......")
        # <idx>_h_matrix.txt store the homography matrix witch transform idx to idx-1
        if os.path.exists(str(idx) + "_h_matrix.txt"):
            H = numpy.loadtxt(str(idx) + "_h_matrix.txt")
            # combine
            output_img = combine_images(src1, src2, H)
            src1 = output_img
        else:
            H = draw_keyPoints_pairs(src1, src2, idx)
            # combine
            output_img = combine_images(src1, src2, H)
            src1 = output_img

    # save
    cv2.imwrite(r'D:\Code\GasTank\demo.jpg', output_img)





