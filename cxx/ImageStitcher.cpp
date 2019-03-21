// OpencvCppTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

vector<Mat> imgs;

int main()
{
	string folder = "../examples/";
	int img_index_start = 1;
	int img_index_end = 2;
	for (int i = img_index_start; i <= img_index_end; i++)
	{
		string file = folder + to_string(i) + ".jpg";
		Mat img = imread(file);
		if (img.empty())
		{
			cout << "Can't read image" << file << endl;
			return -1;
		}
		Mat dst;
		resize(img, dst, Size(1800, 1200));
		imgs.push_back(img);
	}
	/*Mat img1 = imread("1.jpg");
	Mat img2 = imread("2.jpg");


	namedWindow("picture 1", CV_WINDOW_NORMAL);
	namedWindow("picture 2", CV_WINDOW_NORMAL);
	imshow("picture 1", img1);
	imshow("picture 2", img2);

	imgs.push_back(img1);
	imgs.push_back(img2);*/

	//create stitcher
	Stitcher stitcher = Stitcher::createDefault(true);
	//stitching
	Mat panorama;
	Stitcher::Status status = stitcher.stitch(imgs, panorama);
	if (status != Stitcher::OK)
	{
		cout << "can not stitch images, error code = " << int(status) << endl;
		return -1;
	}

	//save and show
	imwrite("result.jpg", panorama);
	namedWindow("result image", CV_WINDOW_NORMAL);
	imshow("result image", panorama);

	waitKey(0);
	return 0;
}