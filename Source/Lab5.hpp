#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct Image {
    //default
    Mat bgr;
    Mat gray;
    Mat quant;

    Mat dct;
};

void quantize(const Mat &src, Mat &dst, int q_level){
    dst = Mat::zeros(src.size(), CV_8U);
    double sko = 0.0;
	int inter = 255 / (q_level - 1);

	// dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {

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
    // cout << "prob = " << prob[0] << endl;
    // cout << "prob = " << prob[1] << endl;
    // cout << "prob = " << prob[255] << endl;
    // sumw
    return prob;
}

double entropy(const vector<double> &prob) {
    double H = 0.0;
    for (int i = 0; i < prob.size(); i++)
        if (prob[i] != 0) H -= prob[i] * log2(prob[i]);
    return H;
}

double redundancy(const double H) {
    return 1 - (H / log2(256));
}

void DCT(Mat &src, Mat &dst) {
    int block_size = 8;
    int delta_h8 = src.rows % block_size;
    int delta_w8 = src.cols % block_size;
    // Mat img_flt = Mat::zeros(src.rows - delta_h8, src.cols - delta_w8, CV_64F);
    // Mat img_flt = Mat::zeros(src.rows - delta_h8, src.cols - delta_w8, CV_8UC1);
    // Mat img_flt = Mat::zeros(src.rows - delta_h8, src.cols - delta_w8, CV_8UC1);
    Mat img_flt;
    src.convertTo(img_flt, CV_64F);

    // Mat img_int;
    // img_flt.convertTo(img_int, CV_8U);
    // imshow("image int", img_int);
    // Mat img_flt = src;
    // copyTo(src, img_flt);
    Mat basisMat = Mat::zeros(block_size, block_size, CV_64F);
    // Mat imageBlock64F;
	// image_block.convertTo(imageBlock64F, CV_64F);
    cout << "Number of blocks row = " << img_flt.rows / block_size << endl;
    cout << "Number of blocks col = " << img_flt.cols / block_size << endl;

    // for (int block_row = 0; block_row < img_flt.rows; block_row += block_size) {
    //     for (int block_col = 0; block_col < img_flt.cols; block_col += block_size) {

    //         for (int row = block_row; row < (block_row + block_size); row++) {
    //             for (int col = block_col; col < (block_col + block_size); col++) {
    //                 // cout<< "got here";
    //                 cout << "row = " << row << "\tcol = " << col << endl;
    //                 // if (block_row + block_size <= src.rows && block_col + block_size <= src.cols) {
    //                     if (row == block_row) basisMat.at<double>(row, col) = 1 / sqrt(block_size);
    //                     else if (row > block_row && row <= (block_row + block_size)) basisMat.at<double>(row, col) = sqrt(2. / block_size) * cos(((CV_PI*row) / block_size)*(col + 0.5));
    //                 // }
    //                 // cout << basisMat.at<double>(row / block_size, col / block_size) << " ";
    //             }
    //         }

    for (int block_row = 0; block_row < img_flt.rows; block_row += block_size) {
        for (int block_col = 0; block_col < img_flt.cols; block_col += block_size) {
            if (block_row + block_size <= img_flt.rows && block_col + block_size <= img_flt.cols) {
                for (int row = 0; row < block_size; row++) {
                    for (int col = 0; col < block_size; col++) {
                        cout << "block_row = " << block_row << "\tblock_col = " << block_col << endl;
                        cout << "row = " << row << "\tcol = " << col << endl;
                        if (row == block_row) basisMat.at<double>(row, col) = 1 / sqrt(block_size);
                        else if (row > block_row) basisMat.at<double>(row, col) = sqrt(2. / block_size) * cos(((CV_PI * row) / block_size) * (col + 0.5));
                    }
                }
            }
        // }
    // }




    // for (int block_row = 0; block_row < src.rows; block_row += block_size) {
    //     for (int block_col = 0; block_col < src.cols; block_col += block_size) {
    //         if (block_row + block_size <= src.rows && block_col + block_size <= src.cols) {
    //             Rect roi_dct_rect(block_col, block_row, block_size, block_size);
    //             Mat roi_dct_flt = img_flt(roi_dct_rect);
    //             Mat roi_dct_int;
    //             roi_dct_flt.convertTo(roi_dct_int, CV_8U);
    //             resize(roi_dct_int, roi_dct_int, Size(400, 400), 0, 0, INTER_NEAREST);
    //             imshow("roi row " + to_string(block_row) + " and col " + to_string(block_col), roi_dct_int);
    //         }
    //     // }
    // // }


            if (block_row + block_size <= src.rows && block_col + block_size <= src.cols) {
                Rect roi_dct_rect(block_col, block_row, block_size, block_size);
                Mat roi_dct_flt = img_flt(roi_dct_rect);
                Mat roi_dct_int;
                roi_dct_flt.convertTo(roi_dct_int, CV_8U);
                resize(roi_dct_int, roi_dct_int, Size(400, 400), 0, 0, INTER_NEAREST);
                // imshow("roi row " + to_string(block_row) + " and col " + to_string(block_col), roi_dct_int);

                Mat DCT = basisMat * roi_dct_flt * basisMat.t();
                cout << "DCT = " << DCT << endl;
                Mat DCT8U;
                DCT.convertTo(DCT8U, CV_8U);
                // imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);
                // basisMat = img_basis(roi8x8);
                Mat img_flt = Mat::zeros(DCT8U.rows, DCT8U.cols, CV_8UC1);
                resize(DCT8U, DCT8U, cv::Size(400, 400), 0, 0, INTER_NEAREST);
                imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);
            }


        // cout << "block row = " << block_row << " block col = " << block_col << endl;
        // Mat basis8u;
        // basisMat.convertTo(basis8u, CV_8U);
        // imshow("basis" + to_string(block_row) + " and col " + to_string(block_col), basis8u);

    
        // Rect roi_dct_rect(block_row, block_col, block_size, block_size);
        // Mat roi_dct_flt = img_flt(roi_dct_rect);
        // Mat roi_dct_int;
        // roi_dct_flt.convertTo(roi_dct_int, CV_8U);
        // // Mat img_dct_img_ = Mat::zeros(roi_dct_flt.rows, roi_dct_flt.cols, CV_8UC1);
        // resize(roi_dct_int, roi_dct_int, cv::Size(400, 400), 0, 0, INTER_NEAREST);
        // imshow("roi row " + to_string(block_row) + " and col " + to_string(block_col), roi_dct_int);



        // cout << roi_dct_flt.size();

        // Mat DCT = basisMat * roi_dct_flt * basisMat.t();
        // Mat DCT8U;
        // DCT.convertTo(DCT8U, CV_8U);
        // // imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);
        // // basisMat = img_basis(roi8x8);
        // Mat img_flt = Mat::zeros(DCT8U.rows, DCT8U.cols, CV_8UC1);
        // resize(DCT8U, DCT8U, cv::Size(400, 400), 0, 0, INTER_NEAREST);
        // imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);
        }
        // for (int row = 0; row < block_size; row++)
        //     for (int col = 0; col < block_size; col++)
        //         img_flt.at<uchar>(row, col) = src.at<uchar>(block + row, block + col);

    }

    // Mat imageBlock64F;
	// image_block.convertTo(imageBlock64F, CV_64F);
    // Mat DCT = basisMat * imageBlock64F * basisMat.t();
    // // std::cout << DCT << std::endl;

	// Mat DCT8U;
	// DCT.convertTo(DCT8U, CV_8U);
    // src = Mat::zeros(DCT8U.rows, DCT8U.cols, CV_8UC1);
    // resize(DCT8U, src, cv::Size(400, 400), 0, 0, INTER_NEAREST);
    // return src;
    return;
}

void lab5(const Mat &img_bgr) {
    Image img;
    img.bgr = img_bgr;
    // resize(img.bgr, img.bgr, Size(300, 300), 0, 0, INTER_CUBIC);
    imshow("image bgr", img.bgr);

    cvtColor(img.bgr, img.gray, COLOR_BGR2GRAY);
    imshow("image gray", img.gray);
    imwrite("../../Images/Lab 5/image gray.jpg", img.gray);

    quantize(img.gray, img.quant, 2);
    // imshow("image quant", img.quant);
    // imwrite("../../Images/Lab 5/image quant.jpg", img.quant);

    vector <double> prob = probality(img.quant);
    double H = entropy(prob);
    cout << "H = " << H << endl;

    double R = redundancy(H);
    cout << "R = " << R << endl;

    DCT(img.gray, img.dct);
    // imshow("image dct", img.dct);
    // imwrite("../../Images/Lab 5/image dct.jpg", img.dct);

    waitKey();
}