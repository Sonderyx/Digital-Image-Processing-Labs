// #include "Lab1.hpp"
// #include "Lab2.hpp"
#include "Lab3.hpp"

int main() {
    // cout << "Hello World!\n";
    string image_path = "../../Images/Orion.png";
    // string image_path = "../../Images/Lena 21.png";
    // string image_path = "../../Images/Lena 70.jpg";
    // string image_path = "../../Images/Berserk.jpg";
    // string image_path = "../../Images/Berserk 2.jpg";
    // string image_path = "../../Images/New York.jpg";

    Mat img_bgr = imread(image_path);
    // lab1(img_bgr);
    // lab2(img_bgr);
    lab3(img_bgr);
    return 0;
}