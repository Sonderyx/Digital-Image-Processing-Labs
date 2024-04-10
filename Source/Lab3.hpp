#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct Image{
    Mat bgr;
    Mat gray;

    Mat gauss3;
    Mat gauss5;

    Mat mosaic3;
    Mat aperture_cor_3;
    Mat median3;

    Mat sharr;
    Mat sobel_vert;


    // Mat full_blocked;
    // Mat selected_block;
    // Mat image_block;
    // int block_size = 8;
};

// Mat getGaussian(const Mat &image, int kernel_size){	


// }

void gauss3(const Mat &input_img, Mat &output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 36; // коэффициент нормировки
    float Fk[3][3] = { {1,4,1}, {4,16,4}, {1,4,1} }; // маска фильтра
    for (int i = 1; i < input_img.cols - 1; i++)
        for (int j = 1; j < input_img.rows - 1; j++) {
        uchar pix_value = input_img.at<uchar>(j, i);
        // далее производим свертку
        float Rez = 0;
        for (int ii = -1; ii <= 1; ii++)
            for (int jj = -1; jj <= 1; jj++) {
                uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                Rez += Fk[ii + 1][jj + 1] * blurred;
            }
        uchar blurred = Rez / k; // осуществляем нормировку
        output_img.at<uchar>(j, i) = blurred;
    }
}

void gauss5(const Mat &input_img, Mat &output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 273; // коэффициент нормировки
    float Fk[5][5] = { 
    {1, 4,  7,  4,  1}, 
    {4, 16, 26, 16, 4}, 
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4,  7,  4,  1}
    }; // маска фильтра
    for (int i = 2; i < input_img.cols - 2; i++)
        for (int j = 2; j < input_img.rows - 2; j++) {
        uchar pix_value = input_img.at<uchar>(j, i);
        // далее производим свертку
        float Rez = 0;
        for (int ii = -2; ii <= 2; ii++)
            for (int jj = -2; jj <= 2; jj++) {
                uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                Rez += Fk[ii + 1][jj + 1] * blurred;
            }
        uchar blurred = Rez / k; // осуществляем нормировку
        output_img.at<uchar>(j, i) = blurred;
    }
}

void mosaic3(const Mat &input_img, Mat &output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 9; // коэффициент нормировки
    float Fk[3][3] = { {1,1,1}, {1,1,1}, {1,1,1} }; // маска фильтра
    for (int i = 1; i < input_img.cols - 1; i+=3)
        for (int j = 1; j < input_img.rows - 1; j+=3) {
        uchar pix_value = input_img.at<uchar>(j, i);
        // далее производим свертку
        float Rez = 0;
        uchar blurred;
        for (int ii = -1; ii <= 1; ii++)
            for (int jj = -1; jj <= 1; jj++) {
                blurred = input_img.at<uchar>(j + jj, i + ii);
                Rez += Fk[ii + 1][jj + 1] * blurred;
                // blurred = Rez / k; // осуществляем нормировку
                // output_img.at<uchar>(j + jj, i + ii) = blurred;
            }

        for (int ii = -1; ii <= 1; ii++)
            for (int jj = -1; jj <= 1; jj++) {
                // uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                // Rez += Fk[ii + 1][jj + 1] * blurred;
                blurred = Rez / k; // осуществляем нормировку
                output_img.at<uchar>(j + jj, i + ii) = blurred;
            }
    }
}


void aperture_cor3(const Mat &input_img, Mat &output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 3; // коэффициент нормировки
    float Fk[3][3] = { {-1,-1,-1}, {-1,11,-1}, {-1,-1,-1} }; // маска фильтра
    for (int i = 1; i < input_img.cols - 1; i++)
        for (int j = 1; j < input_img.rows - 1; j++) {
        uchar pix_value = input_img.at<uchar>(j, i); 
        // далее производим свертку
        float Rez = 0;
        for (int ii = -1; ii <= 1; ii++)
            for (int jj = -1; jj <= 1; jj++) {
                int blurred = input_img.at<uchar>(j + jj, i + ii);
                Rez += Fk[ii + 1][jj + 1] * blurred;
            }
        int blurred = Rez / k; // осуществляем нормировку
        if (blurred < 0) blurred = 0;
        if (blurred > 255) blurred = 255;
        output_img.at<uchar>(j, i) = blurred;
    }
}


// хочу функцию медианный фильтр с маской 3х3
void median3(const Mat &input_img, Mat &output_img){
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 9; // коэффициент нормировки
    float Fk[3][3] = { {1,1,1}, {1,1,1}, {1,1,1} }; // маска фильтра
    
    for (int i = 1; i < input_img.cols - 1; i++)
        for (int j = 1; j < input_img.rows - 1; j++) {
            uchar pix_value = input_img.at<uchar>(j, i);
        // далее производим свертку
            // float Rez = 0;
            vector<uchar> median;
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                median.push_back(input_img.at<uchar>(j + jj, i + ii));
                // median[ii+1][jj+1] = input_img.at<uchar>(jj,ii);
                }
            sort(median.begin(), median.end());
            output_img.at<uchar>(j, i) = median.at(5);
        }
    // cout << "done" << endl;

}

void lab3(const Mat &img_bgr){
    Image img;
    img.bgr = img_bgr;
    imshow("image bgr", img.bgr);

    cvtColor(img_bgr, img.gray, COLOR_BGR2GRAY);
    imshow("image gray", img.gray);

    gauss3(img.gray, img.gauss3);
    imshow("gauss 3", img.gauss3);


    gauss5(img.gray, img.gauss5);
    imshow("gauss 5", img.gauss5);
	// imwrite("../../Images/Lab 2/image gray.jpg", img.gray);

    mosaic3(img.gray, img.mosaic3);
    imshow("mosaic 3", img.mosaic3);

    aperture_cor3(img.gray, img.aperture_cor_3);
    imshow("aperture_cor 3", img.aperture_cor_3);

    median3(img.gray, img.median3);
    imshow("median 3", img.median3);
    cout << "gauss break";
    // // img.block_size = 16;
    // img.full_blocked = getBlocked(img_bgr, img.block_size);
    // imshow("image blocked", img.full_blocked);
    // imwrite("../../Images/Lab 2/image blocked.jpg", img.full_blocked);
    
    waitKey();
}