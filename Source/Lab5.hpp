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
    Mat dct_decoded;
    Mat gray_decoded;
};

Mat getHist(const Mat &src){
    int hist_h = 400, hist_w = 256*3;

	Mat hist = Mat::zeros(1, 256, CV_64FC1);

	for (int i = 0; i < src.cols; i++)
		for (int j = 0; j < src.rows; j++) {
			int r = src.at<unsigned char>(j, i);
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
    Mat img_dct = Mat::zeros(src.rows - delta_h8, src.cols - delta_w8, CV_8UC1);
    // Mat img_flt = Mat::zeros(src.rows - delta_h8, src.cols - delta_w8, CV_8UC1);
    Mat img_flt;
    src.convertTo(img_flt, CV_64F);


    Mat dctMatrix = (Mat_<float>(8, 8) <<
    8, 13, 18, 23, 28, 33, 38, 43,
    13, 18, 23, 28, 33, 38, 43, 48,
    18, 23, 28, 33, 38, 43, 48, 53,
    23, 28, 33, 38, 43, 48, 53, 58,
    28, 33, 38, 43, 48, 53, 58, 63,
    33, 38, 43, 48, 53, 58, 63, 68,
    38, 43, 48, 53, 58, 63, 68, 73,
    43, 48, 53, 58, 63, 68, 73, 78);

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


    for (int block_row = 0; block_row < img_flt.rows; block_row += block_size) {
        for (int block_col = 0; block_col < img_flt.cols; block_col += block_size) {
            if (block_row + block_size <= img_flt.rows && block_col + block_size <= img_flt.cols) {
                for (int row = 0; row < block_size; row++) {
                    for (int col = 0; col < block_size; col++) {
                        // cout << "block_row = " << block_row << "\tblock_col = " << block_col << endl;
                        // cout << "row = " << row << "\tcol = " << col << endl;
                        if (row == block_row) basisMat.at<double>(row, col) = 1 / sqrt(block_size);
                        else if (row > block_row) basisMat.at<double>(row, col) = sqrt(2. / block_size) * cos(((CV_PI * row) / block_size) * (col + 0.5));
                    }
                }
            }

            // вывод блоков дкп по отдельности
            if (block_row + block_size <= src.rows && block_col + block_size <= src.cols) {
                Rect roi_dct_rect(block_col, block_row, block_size, block_size);
                Mat roi_dct_flt = img_flt(roi_dct_rect);
                Mat roi_dct_int;
                roi_dct_flt.convertTo(roi_dct_int, CV_8U);
                resize(roi_dct_int, roi_dct_int, Size(400, 400), 0, 0, INTER_NEAREST);
                // imshow("roi row " + to_string(block_row) + " and col " + to_string(block_col), roi_dct_int);

                Mat DCT = basisMat * roi_dct_flt * basisMat.t();
                // cout << "DCT = " << DCT << endl;

                //-------------------деление на матрицу------------------------
                for (int row = 0; row < block_size; row++) {
                    for (int col = 0; col < block_size; col++) {
                        DCT.at<float>(row, col) /= dctMatrix.at<float>(row, col);
                    }
                }
                //-------------------------------------------------------------

                Mat DCT8U;
                DCT.convertTo(DCT8U, CV_8U);
                // cout << "DCT8U = " << DCT8U << endl;
                // imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);
                // basisMat = img_basis(roi8x8);
                // Mat img_flt = Mat::zeros(DCT8U.rows, DCT8U.cols, CV_8UC1);
                // resize(DCT8U, DCT8U, cv::Size(400, 400), 0, 0, INTER_NEAREST);
                // imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);



                for (int row = 0; row < block_size; row++) {
                    for (int col = 0; col < block_size; col++) {
                        // cout << basisMat.at<double>(row, col) << "\t";
                        uchar Y = DCT8U.at<uchar>(row, col);
                        // cout << "Y = " << Y << endl;
                        img_dct.at<uchar>(block_row + row, block_col + col) = Y;
                        // cout << basisMat.at<double>(row, col) << "\t";
                    }
                    // cout << endl;
                }

            }
            // --------------------------------------





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
    // resize(img_dct, img_dct, cv::Size(400, 400), 0, 0, INTER_NEAREST);
    // resize(img_dct, img_dct, cv::Size(src.rows*2, 600), 0, 0, INTER_NEAREST);
    // imshow("DCT of row"  + to_string(block_row) + " and col " + to_string(block_col), DCT8U);
    // imshow("image dct", img_dct);
    dst = img_dct;
    return;
}

vector<uchar> zigZagBlock(const Mat& block) {
    const uchar n = 8; // Размер блока 8x8
    vector<uchar> output(n * n, 0);
    int i = 0, j = 0;
    bool goingUp = true; // Флаг направления движения

    for (int k = 0; k < n * n; k++) {
        output[k] = block.at<uchar>(i, j); // Использование типа uchar

        if (goingUp) {
            if (j == n - 1) {
                i++; // Двигаемся вниз, если достигли правой границы
                goingUp = false;
            } else if (i == 0) {
                j++; // Двигаемся вправо, если достигли верхней границы
                goingUp = false;
            } else {
                i--;
                j++;
            }
        } else {
            if (i == n - 1) {
                j++; // Двигаемся вправо, если достигли нижней границы
                goingUp = true;
            } else if (j == 0) {
                i++; // Двигаемся вниз, если достигли левой границы
                goingUp = true;
            } else {
                i++;
                j--;
            }
        }
    }

    return output;
}

vector<uchar> zigZagImg(const Mat& src) {
    vector<uchar> output;
    for (int i = 0; i < src.rows; i += 8) {
        for (int j = 0; j < src.cols; j += 8) {
            vector<uchar> block = zigZagBlock(src(Rect(j, i, 8, 8)));
            output.insert(output.end(), block.begin(), block.end());
        }
    }
    return output;
}

// vector<pair<uchar, uchar>> RLECode(const vector<uchar>& data) {
//     vector<pair<uchar, uchar>> encoded; // Вектор для хранения результатов кодирования
//     if (data.empty()) return encoded;

//     uchar count = 1; // Счетчик повторений
//     uchar current = data[0]; // Текущее значение для сравнения

//     // Проходим по данным начиная со второго элемента
//     for (size_t i = 1; i < data.size(); ++i) {
//         if (i % 64 == 0) encoded.push_back(make_pair(255, 255));
//         if (data[i] == current) {
//             count++; // Увеличиваем счетчик, если текущее значение совпадает с предыдущим
//         } else {
//             encoded.push_back(make_pair(count, current)); // Записываем пару (число повторений, значение)
//             current = data[i]; // Обновляем текущее значение
//             count = 1; // Сбрасываем счетчик
//         }

//     }

//     // Добавляем последнюю пару в результат
//     encoded.push_back(make_pair(count, current));

//     // Добавляем код конца блока
//     // encoded.push_back(make_pair(255, 255)); // `255` как символ конца блока

//     return encoded;
// }

vector<pair<uchar, uchar>> RLE_encode(const vector<uchar>& data, int blockSize = 64) {
    vector<pair<uchar, uchar>> encodedData;
    int count = 0;
    int prevValue = (data.empty() ? -1 : data[0]);

    for (int i = 0; i < data.size(); ++i) {
        if (data[i] == prevValue) {
            count++;
        } else {
            if (count > 0) {
                encodedData.emplace_back(count, prevValue);
            }
            prevValue = data[i];
            count = 1;
        }

        // Вставляем код конца блока после обработки каждого блока 8x8
        if ((i + 1) % blockSize == 0) {
            if (count > 0) {
                encodedData.emplace_back(count, prevValue);
                count = 0;
            }
            // Код конца блока
            encodedData.emplace_back(255, 255);
            prevValue = (i + 1 < data.size() ? data[i + 1] : -1);
        }
    }

    // Добавляем оставшиеся данные, если они есть
    if (count > 0 && (data.size() % blockSize) != 0) {
        encodedData.emplace_back(count, prevValue);
    }

    return encodedData;
}

vector<uchar> RLE_decode(const vector<pair<uchar, uchar>>& rle) {
    vector<uchar> decoded;
    for (const auto& pair : rle) {
        if (pair.first == 255 && pair.second == 255) {
            // Окончание блока, но продолжаем обработку, если есть еще данные
            continue;
        }
        for (int i = 0; i < pair.first; i++) {
            decoded.push_back(pair.second);
        }
    }
    return decoded;
}

// Mat zigzagToBlock(const vector<uchar>& zigzag) {
//     Mat block(8, 8, CV_8UC1);
//     int index = 0;
//     for (int i = 0; i < 8; i++) {
//         for (int j = 0; j < 8; j++) {
//             if ((i + j) % 2 == 0) {
//                 // Движение вверх по диагонали
//                 int x = i;
//                 int y = j;
//                 while (x >= 0 && y < 8) {
//                     block.at<float>(x--, y++) = zigzag[index++];
//                 }
//             } else {
//                 // Движение вниз по диагонали
//                 int x = i;
//                 int y = j;
//                 while (x < 8 && y >= 0) {
//                     block.at<float>(x++, y--) = zigzag[index++];
//                 }
//             }
//         }
//     }
//     return block;
// }

// Функция для преобразования зигзаг-последовательности в блок 8x8
Mat zigzagToBlock(const vector<uchar>& zigzag, int startIdx) {
    Mat block(8, 8, CV_32F);  // Используем float для хранения значений
    vector<uchar> indexMap = {
        0,  1,  5,  6, 14, 15, 27, 28,
        2,  4,  7, 13, 16, 26, 29, 42,
        3,  8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    };

    for (int i = 0; i < 64; ++i) {
        int x = i / 8;
        int y = i % 8;
        block.at<float>(x, y) = zigzag[startIdx + indexMap[i]];
    }

    return block;
}

// Функция для создания изображения из всех блоков
Mat createImageFromZigzag(const vector<uchar>& zigzag, int width, int height) {
    CV_Assert(width % 8 == 0 && height % 8 == 0);
    Mat image(height, width, CV_8UC1);

    int blocksPerRow = width / 8;
    int blocksPerColumn = height / 8;
    int index = 0;

    for (int i = 0; i < blocksPerColumn; ++i) {
        for (int j = 0; j < blocksPerRow; ++j) {
            Mat block = zigzagToBlock(zigzag, index);
            block.copyTo(image(Rect(j * 8, i * 8, 8, 8)));
            index += 64;
        }
    }

    return image;
}

// Функция для создания матрицы базисных функций для IDCT
Mat createIDCTMatrix(int size) {
    Mat basis(size, size, CV_32F);
    double scale = sqrt(2.0 / size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            basis.at<float>(j, i) = (i == 0) ? sqrt(1.0 / size) : scale * cos(((2 * j + 1) * i * CV_PI) / (2 * size));
        }
    }
    return basis;
}

// Функция для выполнения обратного ДКП
void IDCT(Mat &src, Mat &dst, const Mat &quantMatrix) {
    int blockSize = 8;
    Mat basis = createIDCTMatrix(blockSize);
    dst = Mat::zeros(src.size(), CV_32F);

    for (int i = 0; i < src.rows; i += blockSize) {
        for (int j = 0; j < src.cols; j += blockSize) {
            Mat block = src(Rect(j, i, blockSize, blockSize));
            Mat temp = Mat::zeros(blockSize, blockSize, CV_32F);

            // Умножаем блок на матрицу квантования
            for (int bi = 0; bi < blockSize; bi++) {
                for (int bj = 0; bj < blockSize; bj++) {
                    temp.at<float>(bi, bj) = block.at<float>(bi, bj) * quantMatrix.at<float>(bi, bj);
                }
            }

            // Выполняем обратное ДКП
            Mat idctBlock = basis.t() * temp * basis;

            // Нормализация для преобразования в CV_8U
            double minVal, maxVal;
            minMaxLoc(idctBlock, &minVal, &maxVal);
            idctBlock.convertTo(idctBlock, CV_32F, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            idctBlock.copyTo(dst(Rect(j, i, blockSize, blockSize)));
        }
    }

    dst.convertTo(dst, CV_8U);  // Конвертация в CV_8U для отображения
}

// void customInverseDCT(Mat& src, Mat& dst, const Mat& quantMatrix) {
//     int blockSize = 8;
//     Mat basis = createIDCTMatrix(blockSize);
//     dst = Mat::zeros(src.size(), CV_32F);

//     double minVal, maxVal;

//     for (int i = 0; i < src.rows; i += blockSize) {
//         for (int j = 0; j < src.cols; j += blockSize) {
//             Mat block = src(Rect(j, i, blockSize, blockSize));
//             Mat temp(blockSize, blockSize, CV_32F);

//             // Умножаем блок на матрицу квантования
//             for (int bi = 0; bi < blockSize; bi++) {
//                 for (int bj = 0; bj < blockSize; bj++) {
//                     temp.at<float>(bi, bj) = block.at<float>(bi, bj) * quantMatrix.at<float>(bi, bj);
//                 }
//             }

//             // Выполняем обратное ДКП
//             Mat idctBlock = basis.t() * temp * basis;

//             minMaxLoc(idctBlock, &minVal, &maxVal);  // Найти мин. и макс. значения
//             cout << "Min value: " << minVal << ", Max value: " << maxVal << endl;

//             // Нормализация
//             idctBlock.convertTo(idctBlock, CV_32F, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
//             idctBlock.copyTo(dst(Rect(j, i, blockSize, blockSize)));
//         }
//     }

//     dst.convertTo(dst, CV_8U);  // Конвертация в CV_8U для отображения
// }

void lab5(const Mat &img_bgr) {
    Image img;
    img.bgr = img_bgr;
    // resize(img.bgr, img.bgr, Size(300, 300), 0, 0, INTER_CUBIC);
    imshow("image bgr", img.bgr);

    cvtColor(img.bgr, img.gray, COLOR_BGR2GRAY);
    imshow("image gray", img.gray);
    imwrite("../../Images/Lab 5/image gray.jpg", img.gray);

    // quantize(img.gray, img.quant, 2);
    // imshow("image quant", img.quant);
    // imwrite("../../Images/Lab 5/image quant.jpg", img.quant);

    vector <double> prob_orig = probality(img.quant);
    double H_orig = entropy(prob_orig);
    cout << "H ref = " << log2(256) << '\t';
    cout << "H orig = " << H_orig << '\t';

    double R = redundancy(H_orig);
    cout << "R orig = " << R << endl;

    DCT(img.gray, img.dct);
    imshow("image dct", img.dct);
    imwrite("../../Images/Lab 5/image dct.jpg", img.dct);

    Mat hist = getHist(img.dct);
    // cout << img.gray.size() << endl;
    // cout << img.dct.size() << endl;
    imshow("Histogram", hist);
    imwrite("../../Images/Lab 5/histogram.jpg", hist);

    vector <double> prob_dct = probality(img.dct);
    double H_dct = entropy(prob_dct);
    cout << "H dct = " << H_dct << '\t';

    double R_dct = redundancy(H_dct);
    cout << "R dct = " << R_dct << endl;

    // Rect block = Rect(285, 401, 8, 8);
    // Mat image_block = img.dct(block);
    // imshow("image block", image_block);
    // cout << image_block << endl;

    // vector <uchar> zig_zag_vec = zigZagBlock(image_block);
    // for (int i = 0; i < zig_zag_vec.size(); i++)
    //     cout << int(zig_zag_vec[i]) << ' ';
    // cout << endl;
    cout << "Total bytes " << img.dct.total() << endl;

    vector <uchar> zig_zag_vec = zigZagImg(img.dct);
    cout << "ZigZag bytes " << zig_zag_vec.size() << endl;

    // for (int i = 0; i < zig_zag_vec.size(); i++)
    //     cout << int(zig_zag_vec[i]) << ' ';
    // cout << endl;

    vector<pair<uchar, uchar>> encoded = RLE_encode(zig_zag_vec);
    // cout << zig_zag_vec.size() << endl;
    // vector<pair<uchar, uchar>> encoded = RLECode(zig_zag_vec);
    // cout << "Encoded RLE: ";
    // for (const auto& pair : encoded) {
    //     cout << int(pair.first) << ";" << int(pair.second) << " ";
    // }
    // cout << endl;

    cout << "Encoded bytes " << encoded.size() * 2 << endl;
    //----------------декодирование---------------------
    vector<uchar> decoded = RLE_decode(encoded);
    cout << "ZigZag decoded bytes " << decoded.size() << endl;

    // img.dct_decoded = zigzagToBlock(decoded, img.dct.rows, img.dct.cols);
    img.dct_decoded = createImageFromZigzag(decoded, img.dct.cols, img.dct.rows);
    // img.dct_decoded = createInverseDCTMatrix(img.dct.rows) * Mat(decoded).reshape(0, img.dct.rows) * createInverseDCTMatrix(img.dct.rows).t();
    imshow("image dct decoded", img.dct_decoded);

    Mat dctMatrix = (Mat_<float>(8, 8) <<
    8, 13, 18, 23, 28, 33, 38, 43,
    13, 18, 23, 28, 33, 38, 43, 48,
    18, 23, 28, 33, 38, 43, 48, 53,
    23, 28, 33, 38, 43, 48, 53, 58,
    28, 33, 38, 43, 48, 53, 58, 63,
    33, 38, 43, 48, 53, 58, 63, 68,
    38, 43, 48, 53, 58, 63, 68, 73,
    43, 48, 53, 58, 63, 68, 73, 78);

    IDCT(img.dct_decoded, img.gray_decoded, dctMatrix);
    // customInverseDCT(img.dct_decoded, img.gray_decoded, dctMatrix);

    imshow("image gray decoded", img.gray_decoded);

    waitKey();
}