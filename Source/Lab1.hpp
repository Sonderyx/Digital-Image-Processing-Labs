#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

Mat getHist(const Mat &image){
    int hist_h = 400, hist_w = 256*3;

	Mat hist = Mat::zeros(1, 256, CV_64FC1);

	for (int i = 0; i < image.cols; i++)
		for (int j = 0; j < image.rows; j++) {
			int r = image.at<unsigned char>(j, i);
			hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
		}
	double m = 0, M = 0;
	minMaxLoc(hist, &m, &M); 
	hist = hist / M;
	Mat hist_img = Mat::zeros(100, 256, CV_8U);
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 100; j++) {
			if (hist.at<double>(0, i) * 100 > j) {
				hist_img.at<unsigned char>(99 - j, i) = 255;
			}
		}
	bitwise_not(hist_img, hist_img);
    resize(hist_img, hist_img, Size(hist_w, hist_h), 0 , 0, INTER_NEAREST);
	return hist_img;
}

Mat getQuant(const Mat &image, int q_level){	
    double sko = 0.0;
	int inter = 255 / (q_level - 1);

	Mat img_quant = Mat::zeros(image.rows, image.cols, CV_8UC1);

	for (int row = 0; row < image.rows; row++){
		for (int col = 0; col < image.cols; col++){

            int Y = image.at<uchar>(row, col);
            
			for (int k = 0; k < q_level; k++){
				if ((Y > inter * k) && (Y <= inter * k + inter / 2)) 		Y = inter * k;
				if ((Y > inter * k + inter / 2) && (Y <= inter * (k + 1)))  Y = inter * (k + 1);
			}
            img_quant.at<uchar>(row, col) = Y;
            sko += (image.at<uchar>(row, col) - img_quant.at<uchar>(row, col)) * (image.at<uchar>(row, col) - img_quant.at<uchar>(row, col));
		}
	}
    sko /= (image.rows*image.cols);
	sko = sqrt(sko);

	cout << "\tsko = " << sko << " \t" << "ass sko = " << inter / sqrt(12) << endl;
	return img_quant;
}

void lab1(const Mat &img_bgr){
    Mat img_gray;
    cvtColor(img_bgr, img_gray, COLOR_BGR2GRAY);

    imshow("image bgr", img_bgr);
    imshow("image gray", img_gray);
	imwrite("../../Images/Lab 1/image gray.jpg", img_gray);

    cout << "for origin: ";
    Mat quant = getQuant(img_gray, 256);
    imshow("Quantization with " + to_string(256) + " levels", quant);
	imwrite("../../Images/Lab 1/Quantization with " + to_string(256) + " levels.jpg", quant);

    Mat hist = getHist(img_gray);
    imshow("Histogram origin", hist);
	imwrite("../../Images/Lab 1/histogram with " + to_string(256) + " levels.jpg", hist);


    for (int q = 2; q < 256; q*=2) {
    cout << "for q = " << q << ": ";
    Mat quant = getQuant(img_gray, q);
    imshow("Quantization with " + to_string(q) + " levels", quant);
	imwrite("../../Images/Lab 1/Quantization with " + to_string(q) + " levels.jpg", quant);
    
    Mat hist = getHist(quant);
    imshow("Histogram with " + to_string(q) + " levels", hist);
	imwrite("../../Images/Lab 1/histogram with " + to_string(q) + " levels.jpg", hist);
    }

    waitKey();
}