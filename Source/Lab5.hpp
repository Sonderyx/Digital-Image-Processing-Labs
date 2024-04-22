#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct Image{
    //default
    Mat bgr;
    Mat gray;
    Mat quant;
};

void quantize(const Mat &src, Mat &dst, int q_level){	
    dst = Mat::zeros(src.size(), CV_8U);
    double sko = 0.0;
	int inter = 255 / (q_level - 1);

	// dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int row = 0; row < src.rows; row++){
		for (int col = 0; col < src.cols; col++){

            int Y = src.at<uchar>(row, col);
            
			for (int k = 0; k < q_level; k++){
				if ((Y > inter * k) && (Y <= inter * k + inter / 2)) 		Y = inter * k;
				if ((Y > inter * k + inter / 2) && (Y <= inter * (k + 1)))  Y = inter * (k + 1);
			}
            dst.at<uchar>(row, col) = Y;
            sko += (src.at<uchar>(row, col) - dst.at<uchar>(row, col)) * (src.at<uchar>(row, col) - dst.at<uchar>(row, col));
		}
	}
    sko /= (src.rows*src.cols);
	sko = sqrt(sko);

	cout << "For q = " << q_level << ": "<< "\tsko = " << sko << " \t" << "ass sko = " << inter / sqrt(12) << endl;
}

vector<double> probality(const Mat &src){
    vector<double> prob(256);
    for (int i = 0; i < src.cols; i++)
        for (int j = 0; j < src.rows; j++) {
            int r = src.at<uchar>(j, i);
            prob[r] += 1.0;
	}

    for (int i = 0; i < prob.size(); i++) prob[i] /= (src.rows*src.cols);
    cout << "prob = " << prob[0] << endl;
    // sumw
    return prob;
}

void DCT(const Mat &image_block, Mat &img_dct){
    int block_size = image_block.rows;
    Mat basisMat = Mat::zeros(block_size, block_size, CV_64F);
    for (int row = 0; row < basisMat.rows;  row++){
        for (int col = 0; col < basisMat.cols; col++){
            if (row == 0) basisMat.at<double>(row, col) = 1 / sqrt(block_size);
            else if (row > 0) basisMat.at<double>(row, col) = sqrt(2. / block_size) * cos(((CV_PI*row) / block_size)*(col + 0.5));
        }
	}

    Mat imageBlock64F;
	image_block.convertTo(imageBlock64F, CV_64F);
    Mat DCT = basisMat * imageBlock64F * basisMat.t();
    // std::cout << DCT << std::endl;

	Mat DCT8U;
	DCT.convertTo(DCT8U, CV_8U);
    img_dct = Mat::zeros(DCT8U.rows, DCT8U.cols, CV_8UC1);
    resize(DCT8U, img_dct, cv::Size(400, 400), 0, 0, INTER_NEAREST);
    // return img_dct;
}


void lab5(const Mat &img_bgr){
    Image img;
    img.bgr = img_bgr;
    // resize(img.bgr, img.bgr, Size(300, 300), 0, 0, INTER_CUBIC);
    imshow("image bgr", img.bgr);

    cvtColor(img.bgr, img.gray, COLOR_BGR2GRAY);
    imshow("image gray", img.gray);
    imwrite("../../Images/Lab 5/image gray.jpg", img.gray);

    quantize(img.gray, img.quant, 2);
    imshow("image quant", img.quant);
    imwrite("../../Images/Lab 5/image quant.jpg", img.quant);

    vector <double> prob = probality(img.quant);

    waitKey();
}