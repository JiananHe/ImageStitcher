import cv2
import numpy

left_key_points = []
right_key_points = []

# flag show the next click
click_flag = -1  # -1 for left, 1 for right
rule_flag = -1  # -1 for left, 1 for right


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global left_key_points, right_key_points, click_flag, rule_flag
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
    img1 = cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)

    src2 = cv2.imread(r'D:\Code\GasTank\examples\example12\p3.jpg')
    src2 = cv2.resize(src2, (1280, 720))
    img2 = cv2.cvtColor(src2, cv2.COLOR_RGB2GRAY)

    # show two pictures
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print("source image1 shape: width=%d, height=%d" % (w1, h1))
    print("source image2 shape: width=%d, height=%d" % (w2, h2))

    vis = numpy.zeros((max(h1, h2), w1 + w2, 3), numpy.uint8)
    vis[:h1, :w1] = src1
    vis[:h2, w1:w1 + w2] = src2

    # callback function, get the coordinates where mouse click
    cv2.namedWindow('image')
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
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

    H, mask = cv2.findHomography(left_key_points, right_key_points, cv2.RANSAC, 5.0)
    print(H)

    # combine
    output_img = combine_images(src2, src1, H)
    # save or display
    cv2.imwrite(r'D:\Code\GasTank\demo.jpg', output_img)
