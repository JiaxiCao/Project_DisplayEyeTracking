#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <time.h>
#include <dshow.h>
#include "windows.h"

using namespace cv;
using namespace std;


int main()
{

    /*VideoCapture v(0+cv::CAP_DSHOW);
    v.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    v.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    v.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));*/
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    while (true) {

        cv::Mat image, imageCopy;
        image = imread("D:/0401-cjx/ahmu/pic/4.png");
        ofstream f("D:/0401-cjx/ahmu/pic/aruco.csv");
        //v >> image;
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        clock_t tim1 = clock();
        cv::aruco::detectMarkers(image, dictionary, corners, ids);//检测靶标
        cout << ids.size() << endl;
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);//绘制检测到的靶标的框
            for (int i = 0; i < corners.size(); ++i)
                cout << corners[i] << ',';
            cout << endl;
            for (int i = 1; i <= 10; ++i) {
                int j = 0;
                for (; j < ids.size(); ++j) {
                    if (ids[j] == i) {
                        f << corners[j][0].x << ',' << corners[j][0].y << ',';
                    }
                }
            }
            f << endl;
        }
        cout << clock() - tim1 << endl;
        pyrDown(imageCopy, imageCopy);
        cv::imshow("out", imageCopy);
        cv::waitKey();

    }

    //cv::Mat image, imageCopy;
    //image = imread("D:/aruco/test.png");
    //image.copyTo(imageCopy);
    //std::vector<int> ids;
    //std::vector<std::vector<cv::Point2f>> corners;
    //clock_t tim1 = clock();
    //cv::aruco::detectMarkers(image, dictionary, corners, ids);//检测靶标
    //cout << ids.size() << endl;
    //if (ids.size() > 0) {
    //    cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);//绘制检测到的靶标的框
    //}
    //cout << clock() - tim1 << endl;
    //cv::imshow("out", imageCopy);
    //cv::waitKey(0);

    /*for (int id = 1; id < 11; ++id) {
        cv::Mat markerImage;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::drawMarker(dictionary, id, 200, markerImage, 1);
        imshow("marker", markerImage);
        imwrite("D:/aruco/generate_marker"+to_string(id)+".png", markerImage);
        waitKey();
    }*/
    

    //cv::Mat imageCopy;
    //markerImage.copyTo(imageCopy);
    //std::vector<int> ids;
    //std::vector<std::vector<cv::Point2f>> corners;
    //cv::aruco::detectMarkers(markerImage, dictionary, corners, ids);//检测靶标
    //cout << ids.size() << endl;
    //if (ids.size() > 0) {
    //    cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);//绘制检测到的靶标的框
    //}
    //cv::imshow("out", imageCopy);
    //cv::waitKey(0);
    return 0;
}
