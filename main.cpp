#include <stdio.h>
#include <iostream>
#include <complex>
#include <math.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define cvQueryHistValue_1D( hist, idx0 ) \  
             ((float)cvGetReal1D( (hist)->bins, (idx0)))  
#define PI 3.1415926

using namespace cv;
using namespace std;

//算术均值、谐波均值、逆谐波均值
Mat filter1(Mat &input, double q, int size, bool geometric) {
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

//中值滤波器 最大值、最小值
//mediean -> choice == 1;
//max -> choice == 2;
// min -> choice == 0;
Mat filter2(Mat &input, int size, int choice) {
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
				if (choice == 0) {
					output.ptr<uchar>(i)[j] = v[0];
				}
				else if (choice == 1) {
					output.ptr<uchar>(i)[j] = v[v.size()/2];
				}
				else {
					output.ptr<uchar>(i)[j] = v[v.size() - 1];
				}
				v.clear();
			}
			else if (channel == 3) {
				//cout << "lalal\n";
				for (int k = 0; k < 3; k ++ ){
					for (int k1 = i - len; k1 <= i + len; k1 ++) {
						for (int k2 = j - len; k2 <= j + len; k2 ++) {
							if (k1 >= 0 && k2 >= 0 && k1 < row && k2 < col) {
								v.push_back(input.at<Vec3b>(k1, k2)[k]);
							}
						}
					}
					sort(v.begin(),v.end());
					if (choice == 0)
						output.at<Vec3b>(i, j)[k] = v[0];
					else if (choice == 1)
						output.at<Vec3b>(i, j)[k] = v[v.size()/2];
					else 
						output.at<Vec3b>(i, j)[k] = v[v.size() - 1];
					//cout << v.size() << endl;
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

IplImage* changeIntoGray(IplImage* input, int choice) {
	IplImage* rImg = cvCreateImage(cvGetSize(input),8,1);
 	IplImage* gImg = cvCreateImage(cvGetSize(input),8,1);
 	IplImage* bImg = cvCreateImage(cvGetSize(input),8,1);
 	cvSplit(input,bImg,gImg,rImg,0);
 	if (choice == 0)
 		return rImg;
 	else if (choice == 1)
 		return gImg;
 	else 
 		return bImg;
}

IplImage* equalize_hist(IplImage **inputImage) {
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inputImage_histogram~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	int size_ = 256;
	float range[2] = {0, 255};
	float* ranges[1] = {range};
	//创建一个维数为1，维数尺寸为256的直方图
	CvHistogram *hist = cvCreateHist(1, &size_, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(inputImage, hist);

	int width_ = 255;
	int height_ = 300;
	int scale_ = 2;
	IplImage *input_image_hist = cvCreateImage(cvSize(width_*scale_, height_), 8, 1);
	//黑底
	cvRectangle(input_image_hist, cvPoint(0, 0), cvPoint(width_*scale_, height_), CV_RGB(0, 0, 0), CV_FILLED);

	//得到直方图的最大值
	float max = 0;
	cvGetMinMaxHistValue(hist, NULL, &max, NULL, NULL);

	vector<float> p;
	float sum = 0;
	for (int i = 0; i <= 255; i ++) {
		//返回相应bin中的值的浮点数
		float hist_value = cvQueryHistValue_1D(hist, i);
		sum += hist_value;
		p.push_back(hist_value);
		//按比率显示高度
		int realHeight_ = cvRound((hist_value / max) * height_);
		CvPoint p1 = cvPoint(i * scale_, height_ - 1);
		CvPoint p2 = cvPoint((i + 1) * scale_ - 1, height_ - realHeight_);
		cvRectangle(input_image_hist, p1, p2, cvScalar(255, 255, 255, 0), CV_FILLED); 
	}
	cout << "sum = " << sum << endl;
	cvShowImage("input_image_hist", input_image_hist);

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~create my outputImage~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	//累积概率分布
	float count = 0;
	for (int i = 0; i <= 255; i ++) {
		count += p[i] / sum;
		p[i] = count;
	}
	//创建一个图
	int result_width = (*inputImage)->width;
	int result_height = (*inputImage)->height;
	IplImage *outputImage= cvCreateImage(cvSize(result_width, result_height), IPL_DEPTH_8U, 1);
	
	//均衡化
	for (int i = 0; i < result_height; i ++)
		for (int j = 0; j < result_width; j ++) {
			double v = cvGetReal2D( (*inputImage), i, j );
			double temp = p[(int)v]*255;
			cvSetReal2D( outputImage, i, j, temp);
		}

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~outputImage_histogram~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	CvHistogram *hist2 = cvCreateHist(1, &size_, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&outputImage, hist2);

	IplImage *output_image_hist = cvCreateImage(cvSize(width_*scale_, height_), 8, 1);
	cvRectangle(output_image_hist, cvPoint(0, 0), cvPoint(width_*scale_, height_), CV_RGB(0, 0, 0), CV_FILLED);

	float max2 = 0;
	cvGetMinMaxHistValue(hist2, NULL, &max2, NULL, NULL);
	for (int i = 0; i <= 255; i ++) {
		float hist_value = cvQueryHistValue_1D(hist2, i);
		int realHeight_ = cvRound((hist_value / max2) * height_);
		CvPoint p1 = cvPoint(i * scale_, height_ - 1);
		CvPoint p2 = cvPoint((i + 1) * scale_ - 1, height_ - realHeight_);
		cvRectangle(output_image_hist, p1, p2, cvScalar(255, 255, 255, 0), CV_FILLED); 
	}
	cvShowImage("output_image_hist", output_image_hist);

	return outputImage;
}

IplImage* changeIntoRebuild(IplImage** rImg, IplImage** gImg, IplImage** bImg) {

	IplImage *average= cvCreateImage(cvSize((*rImg)->width, (*rImg)->height), IPL_DEPTH_8U, 1);
	for (int i = 0; i < (*rImg)->height; i ++)
		for (int j = 0; j < (*rImg)->width; j ++) {
			double v1 = cvGetReal2D( (*rImg), i, j );
			double v2 = cvGetReal2D( (*gImg), i, j );
			double v3 = cvGetReal2D( (*bImg), i, j );
			double temp = (v1 + v2 + v3) / 3;
			cvSetReal2D( average, i, j, temp);
		}
	//cvShowImage("average", average);

	int size_ = 256;
	float range[2] = {0, 255};
	float* ranges[1] = {range};
	//创建维数为1，维数尺寸为256的直方图
	CvHistogram *histAverage = cvCreateHist(1, &size_, CV_HIST_ARRAY, ranges, 1);
	cvCalcHist(&average, histAverage);

	vector<float> V;
	for (int i = 0; i <= 255; i ++) {
		//返回相应bin中的值的浮点数
		float hist_value = cvQueryHistValue_1D(histAverage, i);
		V.push_back(hist_value);
	}

	float sum = (*rImg)->width * (*rImg)->height;
	//累积概率分布
	float count = 0;
	for (int i = 0; i <= 255; i ++) {
		count += V[i] / sum;
		V[i] = count;
	}

	//创建3个图
	int result_width = (*rImg)->width;
	int result_height = (*rImg)->height;
	IplImage *outputR= cvCreateImage(cvSize(result_width, result_height), IPL_DEPTH_8U, 1);
	IplImage *outputG= cvCreateImage(cvSize(result_width, result_height), IPL_DEPTH_8U, 1);
	IplImage *outputB= cvCreateImage(cvSize(result_width, result_height), IPL_DEPTH_8U, 1);
	
	//映射
	for (int i = 0; i < result_height; i ++)
		for (int j = 0; j < result_width; j ++) {
			double v = cvGetReal2D( (*rImg), i, j );
			double temp = V[(int)v]*255;
			cvSetReal2D( outputR, i, j, temp);

			v = cvGetReal2D( (*gImg), i, j );
			temp = V[(int)v]*255;
			cvSetReal2D( outputG, i, j, temp);

			v = cvGetReal2D( (*bImg), i, j );
			temp = V[(int)v]*255;
			cvSetReal2D( outputB, i, j, temp);
		}

	IplImage *output = cvCreateImage(cvGetSize(outputB),8,3);
    	cvMerge(outputB, outputG, outputR, 0, output);

	return output;
} 


int main(int argc, char** argv )
{
	//读入input.png
	//对应part1 part2 part3 分别输入 task_1.png task_2.png 07.png
	
	//~~~~~~~~~~~~~~~~~~~~~~~~part1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
	
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~part2-2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//part2-2 add addGaussian noise
	/*
	Mat image2;
	image2 = imread( "task_2.png");
	if ( !image2.data )
	{
		printf("No image2 data \n");
		return -1;
	}
	namedWindow("Display input Image", WINDOW_AUTOSIZE );
	imshow("Display input Image", image2);
	
	Mat part2 = addGaussian(image2, 0, 40);
	namedWindow("Display addGaussian Image", WINDOW_AUTOSIZE );
	imshow("Display addGaussian Image", part2);

	Mat part2_2 = filter1(part2, 0, 9, false);
	namedWindow("Display arithmetic mean filtering Image", WINDOW_AUTOSIZE );
	imshow("Display arithmetic mean filtering Image", part2_2);

	Mat part2_3 = filter1(part2, 0, 9, true);
	namedWindow("Display GeometricMeanFilter Image", WINDOW_AUTOSIZE );
	imshow("Display GeometricMeanFilter Image", part2_3);

	Mat part2_4 = filter2(part2, 9, 1);
	namedWindow("Display median filtering  Image", WINDOW_AUTOSIZE );
	imshow("Display median filtering  Image", part2_4);
	*/

	//~~~~~~~~~~~~~~~~~~~~~~~~~part2-3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//part2-3 Add salt noise
	/*
	Mat image3;
	image3 = imread( "task_2.png");
	if ( !image3.data )
	{
		printf("No image3 data \n");
		return -1;
	}
	namedWindow("Display input Image", WINDOW_AUTOSIZE );
	imshow("Display input Image", image3);

	Mat part3 = addSaltPepper(image3, 0.2, true);
	namedWindow("Display addSaltPepper Image", WINDOW_AUTOSIZE );
	imshow("Display addSaltPepper Image", part3);

	Mat part3_2 = filter1(part3, -1, 9, false);
	namedWindow("Display part3 harmonic mean filter Image", WINDOW_AUTOSIZE );
	imshow("Display part3 harmonic mean filter Image", part3_2);

	Mat part3_3 = filter1(part3, -1.5, 9, false);
	namedWindow("Display part3_3 contraharmonic Q= -1.5 Image", WINDOW_AUTOSIZE );
	imshow("Display part3_3 contraharmonic Q= -1.5 Image", part3_3);
	*/

	//~~~~~~~~~~~~~~~~~~~~~~~~~part2-4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//part2-4 Add salt and pepper noise
	/*
	Mat image4;
	image4 = imread( "task_2.png");
	if ( !image4.data )
	{
		printf("No image4 data \n");
		return -1;
	}
	namedWindow("Display input Image", WINDOW_AUTOSIZE );
	imshow("Display input Image", image4);

	Mat part4 = addSaltPepper(image4, 0.2, true); // add salt noise
	part4 = addSaltPepper(part4, 0.2, false); // add pepper noise
	namedWindow("Display part4 addSaltPepper Image", WINDOW_AUTOSIZE );
	imshow("Display part4 addSaltPepper Image", part4);

	Mat part4_2 = filter1(part4, 0, 9, false);
	namedWindow("Display part4 arithmetic mean filtering Image", WINDOW_AUTOSIZE );
	imshow("Display part4 arithmetic mean filtering Image", part4_2);

	Mat part4_3 = filter1(part4, 0, 9, true);
	namedWindow("Display part4 GeometricMeanFilter Image", WINDOW_AUTOSIZE );
	imshow("Display part4 GeometricMeanFilter Image", part4_3);

	Mat part4_4 = filter2(part4, 3, 1);
	namedWindow("Display part4 mediean filtering Image", WINDOW_AUTOSIZE );
	imshow("Display part4 mediean filtering Image", part4_4);
	*/

	//~~~~~~~~~~~~~~~~~~~~~~~~~~Part3-1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IplImage *input = cvLoadImage("07.png", -1);
	cvNamedWindow("input");  
    	cvShowImage("input",input);
	//cout << input->nChannels << endl;
 	
    	IplImage *rImg = changeIntoGray(input, 0);
    	cvNamedWindow("rImg");  
    	cvShowImage("rImg",rImg);
    	IplImage *equalize_rImg = equalize_hist(&rImg);
	cvNamedWindow("equalize_rImg");  
    	cvShowImage("equalize_rImg",equalize_rImg);
    	
    	IplImage *gImg = changeIntoGray(input, 1);
    	//cvNamedWindow("gImg");  
    	//cvShowImage("gImg",gImg);
    	IplImage *equalize_gImg = equalize_hist(&gImg);
	//cvNamedWindow("equalize_gImg");  
    	//cvShowImage("equalize_gImg",equalize_gImg);
    	
    	IplImage *bImg = changeIntoGray(input, 1);
    	//cvNamedWindow("bImg");  
    	//cvShowImage("bImg",bImg);
    	IplImage *equalize_bImg = equalize_hist(&bImg);
	//cvNamedWindow("equalize_bImg");  
    	//cvShowImage("equalize_bImg",equalize_bImg);

    	IplImage *rebuild = cvCreateImage(cvGetSize(input),8,3);
    	cvMerge(equalize_bImg, equalize_gImg, equalize_rImg, 0, rebuild);
    	cvNamedWindow("rebuild");  
    	cvShowImage("rebuild",rebuild);

    	
    	IplImage *rebuild2 = changeIntoRebuild(&rImg, &gImg, &bImg);
    	cvNamedWindow("rebuild2");  
    	cvShowImage("rebuild2",rebuild2);

    	waitKey(0);
  	return 0;
}
