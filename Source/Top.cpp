// #include "Lab1.hpp"
// #include "Lab2.hpp"
// #include "Lab3.hpp"
// #include "Lab4.hpp"
#include "Lab5.hpp"

int main() {
    // string image_path = "../../Images/Orion.png";
    // string image_path = "../../Images/Lena 21.png";
    // string image_path = "../../Images/Lena 70.jpg";
    // string image_path = "../../Images/Berserk.jpg";
    // string image_path = "../../Images/Berserk 2.jpg";
    // string image_path = "../../Images/New York.jpg";
    // string image_path = "../../Images/Dash.jpg";
    // string image_path = "../../Images/FP1.png";
    // string image_path = "../../Images/Shrek.png";
    // string image_path = "../../Images/Morph1.png";
    // string image_path = "../../Images/Plane.jpg";
    // string image_path = "../../Images/Levitan.jpg";
    string image_path = "../../Images/test.jpg";

    Mat img_bgr = imread(image_path);
    // lab1(img_bgr);
    // lab2(img_bgr);
    // lab3(img_bgr);
    // lab4(img_bgr);
    lab5(img_bgr);
    return 0;
}