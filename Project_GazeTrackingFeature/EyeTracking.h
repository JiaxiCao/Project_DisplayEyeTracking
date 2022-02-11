#pragma once
#include <iostream>
#include <opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <string>
#include <fstream>
#include "windows.h"
#include "dshow.h"

using namespace std;
using namespace cv;

/*Nerual Network paramaters*/
extern double
xmaxarray_glint[24], xminarray_glint[24], ymaxarray_glint[2], yminarray_glint[2],
W1_linarray[48], B1_linarray[2];

class EyeTracker
{
public:
	bool Init_Camera(int idx1, int idx2, int idx3, int idx4, int idx_scene);
	bool Init_Camera(string url1, string url2, string url3, string url4, string url_scene, bool iscalib, string outfile="");
	void Init();

	void do_process();

private:
	void NerualNetwork_Init();
	Mat xmax_glint, xmin_glint;
	Mat W1_lin, B1_lin;
	bool is_calib = false;

	VideoCapture eye1, eye2, eye3, eye4, scene;
	Mat eye1_img, eye2_img, eye3_img, eye4_img;
	Mat scene_img;
	Point coarse_center1, coarse_center2, coarse_center3, coarse_center4;
	Mat FRST(Mat img);
	Point findCoarseCenterFRST(Mat img);

	void edgeConnectivity(Mat canny, vector<vector<Point>>& contours);
	void qsort(vector<int>& data, vector<int>& idx, int h, int t);
	RotatedRect pupilReconstruction(vector<vector<Point>> edges, Mat img, Point coarsecenter);
	void StarBurst(Mat canny_img, Point coarsecenter, int distance, vector<Vec2i>& pupiledge, vector<int>& R);
	RotatedRect pupilfitStarBurst(vector<Vec2i>& pupiledge, vector<int>& R, int distance);
	float EllipDifferiential(Mat img, RotatedRect ellip);
	RotatedRect pupilDetect(Mat canny_img, Mat img, Point coarsecenter, int distance);
	void findGlint(Mat img, vector<Point>& glints, int thresh, int kernel_size, Point pupil_center);
	int Dnum;
	Point2f point_esti;
	Point2f onep_compensation = Point2f(0, 0);

	ofstream fresult;

	Point2f FilterGaze(Point2f gazept);
	vector<Point2f> gaze_seq;
	void GroundTruthDetect(Mat scene_);
	Point2f groundtruth_pt;
	vector<Point3f> calib_eye;
	vector<Point2f> calib_gaze;
	Mat calibgaze_mat;
	void OnePointCompensation(Mat scene_, Point2f gaze_point);

	void arucoDetect(Mat scene_, std::vector<int>& ids, std::vector<std::vector<cv::Point2f>>& corners);

	VideoWriter writer;
};