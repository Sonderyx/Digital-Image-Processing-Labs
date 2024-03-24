#include "Lab1.hpp"
#include "Lab2.hpp"

int main() {
    // cout << "Hello World!\n";
    // string image_path = "../../Images/Orion.png";
    string image_path = "../../Images/Lenna.PNG";
    // string image_path = "../../Images/Berserk.jpg";
    // string image_path = "../../Images/New York.jpg";
    Mat img_bgr = imread(image_path), img_gray;
    lab1(img_bgr);
    return 0;
}