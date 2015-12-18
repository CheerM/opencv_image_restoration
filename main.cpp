#include <stdio.h>
#include <iostream>
#include <complex>
#include <math.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define PI 3.1415926

using namespace cv;
using namespace std;

//算术均值、谐波均值、逆谐波均值
Mat filter1(Mat &input, int q, int size, bool geometric) {
	int row = input.rows;
	int col = input.cols;
	int channel = input.channels();
	Mat output = input.clone();
	double temp1 = 0; //分子
	double temp2 = 0; //分母
	double temp = 1; ////几何均值滤波器_累乘
	int len = size / 2; //用以判断是3*3 还是9*9大小的滤波器，再做循环
	for (int i = 0; i < row; i ++) {
		for (int j = 0; j < col; j ++) {
			if (channel == 1) {
				for (int k1 = i - len; k1 <= i + len; k1 ++) {
					for (int k2 = j - len; k2 <= j + len; k2 ++) {
						if (k1 >= 0 && k2 >= 0 && k1 < row && k2 < col) {
							if (!geometric) {
								temp1 += pow(input.ptr<uchar>(k1)[k2], q+1);
								temp2 += pow(input.ptr<uchar>(k1)[k2], q);
							}
							else {
								temp *= input.ptr<uchar>(k1)[k2];
							}
						}
					}
				}
				if (!geometric) {
					output.ptr<uchar>(i)[j] =saturate_cast<uchar>(temp1 / temp2);
					temp1 = 0;
					temp2 = 0;
				}
				else {
					output.ptr<uchar>(i)[j] = saturate_cast<uchar>(pow(temp, 1/(size * size + 0.0)));
					temp = 1;
				}
			}
			else if (channel == 3) {
				for (int k = 0; k < 3; k ++ ){
					for (int k1 = i - len; k1 <= i + len; k1 ++) {
						for (int k2 = j - len; k2 <= j + len; k2 ++) {
							if (k1 >= 0 && k2 >= 0 && k1 < row && k2 < col) {
								if (!geometric) {
									temp1 += pow(input.at<Vec3b>(k1, k2)[k], q+1);
									temp2 += pow(input.at<Vec3b>(k1, k2)[k], q);
								}
								else {
									temp *= input.at<Vec3b>(k1, k2)[k];
								}
							}
						}
					}
					if (!geometric) {
						output.at<Vec3b>(i, j)[k] =saturate_cast<uchar>(temp1 / temp2);
						temp1 = 0;
						temp2 = 0;
					}
					else {
						output.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(pow(temp, 1/(size * size + 0.0)));
						temp = 1;
					}
				}
			}
		}
	}
	return output;
}

//中值滤波器
Mat MedianFilter(Mat &input, int size) {
	int row = input.rows;
	int col = input.cols;
	int channel = input.channels();
	Mat output = input.clone();
	int len = size / 2;
	vector<uchar> v;
	for (int i = 0; i < row; i ++) {
		for (int j = 0; j < col; j ++) {
			if (channel == 1) {
				for (int k1 = i - len; k1 <= i + len; k1 ++) {
					for (int k2 = j - len; k2 <= j + len; k2 ++) {
						if (k1 >= 0 && k2 >= 0 && k1 < row && k2 < col) {
							v.push_back(input.ptr<uchar>(k1)[k2]);
						}
					}
				}
				sort(v.begin(),v.end());
				output.ptr<uchar>(i)[j] = v[v.size()/2];
				v.clear();
			}
			else if (channel == 3) {
				//cout << "lalal\n";
				for (int k = 0; k < 3; k ++ ){
					for (int k1 = i - len; k1 <= i + len; k1 ++) {
						for (int k2 = j - len; k2 <= j + len; k2 ++) {
							if (k1 >= 0 && k2 >= 0 && k1 < row && k2 < col) {
								v.push_back(output.at<Vec3b>(k1, k2)[k]);
							}
						}
					}
					sort(v.begin(),v.end());
					output.at<Vec3b>(i, j)[k] = v[v.size()/2];
					v.clear();
				}
			}
		}
	}
	return output;
}

double Random() {
    	double u1 = (double)(rand()/(double)RAND_MAX);
	double u2 = (double)(rand()/(double)RAND_MAX);
	double Z = sqrt(-2 * log(u2)) * cos(2 * PI * u1);
    	return Z;
}

Mat addGaussian(Mat &input, int mean, int standard) {
	srand((unsigned)time(NULL));
	int row = input.rows;
	int col = input.cols;
	int channel = input.channels();
	Mat output = input.clone();
	for (int i = 0 ; i < row; i ++) {
		for (int j = 0; j < col; j ++) {
			double result = Random() * standard + mean;
			if (channel == 3) {
				for (int k = 0; k < 3; k ++)
					output.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(result + output.at<Vec3b>(i, j)[k]);
			}
			else {
				output.ptr<uchar>(i)[j] = saturate_cast<uchar>(result + output.ptr<uchar>(i)[j]);
			}
		}
	}
	return output;
}

Mat addSaltPepper(Mat &input, double probability, bool salt) {
	int row = input.rows;
	int col = input.cols;
	int channel = input.channels();
	Mat output = input.clone();
	int u1;
	int u2;
	int count = (int)(probability * row * col);
	while (count -- ){
		u1 = rand() % row;
		u2 = rand() % col;
		if (salt) {
			for (int k = 0; k < 3; k ++)
				output.at<Vec3b>(u1, u2)[k]= 255;
		}
		else {
			for (int k = 0; k < 3; k ++)
				output.at<Vec3b>(u1, u2)[k]= 0;
		}
	}
	return output;
}

int main(int argc, char** argv )
{
	//读入input.png
	//对应part1 part2 part3 分别输入 task_1.png task_2.png 07.png
	

	//part1 contra-harmonic mean filter
	/*
	Mat image1;
	image1 = imread( "task_1.png", -1 );
	if ( !image1.data )
	{
		printf("No image1 data \n");
		return -1;
	}
	namedWindow("Display input Image", WINDOW_AUTOSIZE );
	imshow("Display input Image", image1);

	Mat part1 = filter1(image1, 0, 3, false);
	namedWindow("Display part1 Image", WINDOW_AUTOSIZE );
	imshow("Display part1 Image", part1);
	*/
	
	//part2-1 add addGaussian noise
	Mat image2;
	image2 = imread( "task_2.png");
	if ( !image2.data )
	{
		printf("No image2 data \n");
		return -1;
	}
	//namedWindow("Display input Image", WINDOW_AUTOSIZE );
	//imshow("Display input Image", image2);
	
	Mat part2 = addGaussian(image2, 0, 40);
	namedWindow("Display addGaussian Image", WINDOW_AUTOSIZE );
	imshow("Display addGaussian Image", part2);

	Mat part2_2 = filter1(part2, 0, 9, false);
	namedWindow("Display arithmetic mean filtering Image", WINDOW_AUTOSIZE );
	imshow("Display arithmetic mean filtering Image", part2_2);

	Mat part2_3 = filter1(part2, 0, 9, true);
	namedWindow("Display GeometricMeanFilter Image", WINDOW_AUTOSIZE );
	imshow("Display GeometricMeanFilter Image", part2_3);

	Mat part2_4 = MedianFilter(part2, 9);
	namedWindow("Display median filtering  Image", WINDOW_AUTOSIZE );
	imshow("Display median filtering  Image", part2_4);
	/*
	Mat part3 = addSaltPepper(image2, 0.2, false);
	namedWindow("Display addSaltPepper Image", WINDOW_AUTOSIZE );
	imshow("Display addSaltPepper Image", part3);*/

    	waitKey(0);
  	return 0;
}