#include"EyeTracking.h"

bool EyeTracker::Init_Camera(int idx1, int idx2, int idx3, int idx4, int idx_scene)
{
	eye1.open(idx1 + cv::CAP_DSHOW);
	eye2.open(idx2 + cv::CAP_DSHOW);
	eye3.open(idx3 + cv::CAP_DSHOW);
	eye4.open(idx4 + cv::CAP_DSHOW);
	scene.open(idx_scene + cv::CAP_DSHOW);

	if (!eye1.isOpened() || !eye2.isOpened() || !eye3.isOpened() || !eye4.isOpened() || !scene.isOpened())
	{
		std::cout << "Can not open camera.\n";
		return false;
	}

	eye1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	eye1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	eye1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
	eye2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	eye2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	eye2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
	eye3.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	eye3.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	eye3.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
	eye4.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	eye4.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	eye4.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));

	scene.set(cv::CAP_PROP_AUTOFOCUS, 0);
	scene.set(CAP_PROP_FOCUS, 5);
	scene.set(cv::CAP_PROP_FRAME_WIDTH, 1920);						/*Set frame size*/
	scene.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	scene.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));	/*Default YUV2 codecs fps is 10, MJPEG codecs fps is 50+*/

	return true;
}
bool EyeTracker::Init_Camera(string url1, string url2, string url3, string url4, string url_scene, bool iscalib, string outfile)
{
	eye1.open(url1);
	eye2.open(url2);
	eye3.open(url3);
	eye4.open(url4);
	scene.open(url_scene);

	if (!eye1.isOpened() || !eye2.isOpened() || !eye3.isOpened() || !eye4.isOpened() || !scene.isOpened())
	{
		cout << "Can not open camera.\n";
		return false;
	}
	is_calib = iscalib;
	if (outfile.length() > 0) fresult.open(outfile);
	else fresult.open("result.csv");

	cout << "Project 1 open video success." << endl;
	return true;
}
void EyeTracker::Init()
{
	Dnum = 180;

	NerualNetwork_Init();
}
void EyeTracker::NerualNetwork_Init()
{
	/*linear gaze model*/
	xmax_glint = Mat(24, 1, CV_64F, xmaxarray_glint);
	xmin_glint = Mat(24, 1, CV_64F, xminarray_glint);
	W1_lin = Mat(2, 24, CV_64F, W1_linarray);
	B1_lin = Mat(2, 1, CV_64F, B1_linarray);
}
Mat EyeTracker::FRST(Mat img)
{
	Mat frst;
	Mat sobelx, sobely;

	//Sobel edge detect
	Sobel(img, sobelx, -1, 1, 0);
	Sobel(img, sobely, -1, 0, 1);
	Mat G = sobelx.mul(sobelx) + sobely.mul(sobely);
	double maxG;
	minMaxLoc(G, NULL, &maxG, NULL, NULL);

	//Gradient magnitude thresholding
	for (int i = 0; i < G.rows; i++)
		for (int j = 0; j < G.cols; j++)
		{
			if (G.at<double>(i, j) < 0.04 * maxG)
				G.at<double>(i, j) = 0;
			else
				G.at<double>(i, j) = sqrt(G.at<double>(i, j));
		}

	//RST
	Mat S = Mat(G.rows, G.cols, CV_64F, Scalar::all(0));
	for (int N = 8; N < 25; N++)
	{
		Mat O = Mat(G.rows, G.cols, CV_64F, Scalar::all(0));
		Mat M = Mat(G.rows, G.cols, CV_64F, Scalar::all(0));
		for (int i = 10; i < G.rows - 10; i++)
			for (int j = 10; j < G.cols - 10; j++)
			{
				if (G.at<double>(i, j) > 0)
				{
					int y = (int)(i + N * sobely.at<double>(i, j) / G.at<double>(i, j));
					int x = (int)(j + N * sobelx.at<double>(i, j) / G.at<double>(i, j));
					if (x > 0 && y > 0 && x < G.cols && y < G.rows)
					{
						O.at<double>(y, x) += 1;
						M.at<double>(y, x) += G.at<double>(i, j);
					}
				}

			}
		S = S + O.mul(M);
	}
	frst = S / (25 - 8 + 1);
	return frst;
}
Point EyeTracker::findCoarseCenterFRST(Mat img)
{
	Point center;
	Mat gray;
	pyrDown(img, gray);
	for (int i = 0; i < 2 - 1; i++)
		pyrDown(gray, gray);
	//1. enhance contrast
	//CLAHE enhance contrast
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(3);
	clahe->apply(gray, gray);
	//imshow("gray", gray);
	//cvWaitKey(0);
	//Convert to double type
	gray.convertTo(gray, CV_64F);
	gray = gray / 255;

	//2. fast radial symmetric transform
	//Calculate pupil map
	Mat I_ = gray.clone();
	I_ = 0.4 * abs(I_ - 0.5) + 0.8;
	Mat Ie;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(I_, I_, MORPH_DILATE, kernel);
	morphologyEx(gray, Ie, MORPH_ERODE, kernel);
	Mat Ie_mean, Ie_stddev;
	meanStdDev(Ie, Ie_mean, Ie_stddev);
	Mat pupil_map = I_ / (Ie + Ie_mean.at<double>(0, 0));//[2018 Martinikorena]

	//Fast radial symmetry transform
	Mat Sp = FRST(pupil_map);
	Mat Si = FRST(1 - gray);
	Mat S = Sp + Si;
	GaussianBlur(S, S, Size(7, 7), 7);
	//imshow("S", S);
	double maxS;
	minMaxLoc(S, NULL, &maxS, NULL, &center);
	if (maxS < 10)
	{
		center.x = -1;
		center.y = -1;
	}
	for (int i = 0; i < 2; i++)
		center = center * 2;
	return center;
}
void EyeTracker::edgeConnectivity(Mat canny, vector<vector<Point>>& contours) {
	vector<Vec4i> hierarchy;
	findContours(canny, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (int i = contours.size() - 1; i > -1; --i) {
		if (contours[i].size() < 100) contours.erase(contours.begin() + i);
	}
}
void EyeTracker::qsort(vector<int>& data, vector<int>& idx, int h, int t) {
	if (h >= t) return;
	int tmp = data[h];
	int idxtmp = idx[h];
	int i = h, j = t;
	while (i < j) {
		while (i < j && data[j] <= tmp) --j;
		while (i < j && data[i] >= tmp) ++i;
		if (i < j) {
			int exchange = data[i];
			data[i] = data[j];
			data[j] = exchange;
			exchange = idx[i];
			idx[i] = idx[j];
			idx[j] = exchange;
		}
	}
	data[h] = data[i];
	idx[h] = idx[i];
	data[i] = tmp;
	idx[i] = idxtmp;
	qsort(data, idx, h, i - 1);
	qsort(data, idx, i + 1, t);
}
void EyeTracker::StarBurst(Mat canny_img, Point coarsecenter, int distance, vector<Vec2i>& pupiledge, vector<int>& R) {
	if (coarsecenter.x < 0 | coarsecenter.y < 0) return;
	int search_len = int(2 * distance) + 1;
	for (int i = 0; i < Dnum + 1; i++)
	{
		for (int j = 0; j < search_len; j++)
		{
			int u = coarsecenter.x + j * cos(i * (360 / Dnum) * CV_PI / 180);
			int v = coarsecenter.y + j * sin(i * (360 / Dnum) * CV_PI / 180);
			if (v > canny_img.rows - 1 || v <= 0 || u > canny_img.cols - 1 || u <= 0 || j == search_len - 1)
			{
				pupiledge.push_back(Vec2i(-1, -1));
				R.push_back(0);
				break;
			}
			if (canny_img.at<uchar>(v, u) == 255)
			{
				pupiledge.push_back(Vec2i(u, v));
				R.push_back(j);
				break;
			}
		}
	}
}
RotatedRect EyeTracker::pupilfitStarBurst(vector<Vec2i>& pupiledge, vector<int>& R, int distance) {
	RotatedRect EllipseImg;
	int d = int(ceil(distance * 2 * CV_PI / 180)) - 1;
	int k = 1;
	int start = R[0];
	vector<int> templist;
	vector<int> maxlist;
	vector<int> secondlist;
	templist.push_back(0);

	for (int i = 1; i < R.size(); i++)
	{
		if (R[i] == 0)
		{
			k++;
			continue;
		}
		if (k > 2) k = 2;
		if ((R[i] - start > -(d * k + 1)) && (R[i] - start < (d * k + 1)))
		{
			templist.push_back(i);
			k = 1;
		}
		else
		{
			if (i + 1 > Dnum) break;
			if ((R[i + 1] - start > -(d * k + 1)) && (R[i + 1] - start < (d * k + 1)))
			{
				templist.push_back(i + 1);
				i++;
			}
			else
			{
				if (templist.size() > maxlist.size())
				{
					secondlist = maxlist;
					maxlist = templist;
					templist.clear();
				}
				else
				{
					if (templist.size() > secondlist.size())
					{
						secondlist = templist;
						templist.clear();
					}
					else templist.clear();
				}
			}
		}
		start = R[i];
	}
	if (templist.size() > maxlist.size())
	{
		secondlist = maxlist;
		maxlist = templist;
	}
	else
	{
		if (templist.size() >= secondlist.size())
			secondlist = templist;
	}
	if (secondlist.size() > 15) maxlist.insert(maxlist.end(), secondlist.begin(), secondlist.end());
	templist.clear();
	int lendis = R.size();
	if (maxlist[0] == 0 && maxlist[maxlist.size() - 1] != Dnum)
	{
		start = R[lendis - 1];
		for (int i = 2; i < lendis; i++)
		{
			if (R[lendis - i] == 0)
			{
				k++;
				continue;
			}
			if (R[lendis - i] - start > -(d * k + 1) && R[lendis - i] - start < (d * k + 1))
			{
				templist.push_back(lendis - i);
				k = 1;
			}
			else
			{
				if (R[lendis - i - 1] - start > -(d * k + 1) && R[lendis - i - 1] - start < (d * k + 1))
				{
					templist.push_back(lendis - i - 1);
					i++;
				}
				else break;
			}
		}
	}
	vector<Vec2i> npedge;
	maxlist.insert(maxlist.end(), templist.begin(), templist.end());
	if (maxlist.size() < 50)//边缘点数过少，不进行拟合
		return EllipseImg;
	for (int i = 0; i < maxlist.size(); i++)
		if (pupiledge[maxlist[i]][0] > 0)
		{
			npedge.push_back(Vec2i(pupiledge[maxlist[i]][0], pupiledge[maxlist[i]][1]));
		}
}
float EyeTracker::EllipDifferiential(Mat img, RotatedRect ellip) {
	float ellip_diff = 0;
	int ellip_cnt = 0;
	for (int i = 0; i < 360; ++i) {
		int x_in = ellip.center.x + (ellip.size.width / 2.0 - 2) * cos(i * CV_PI / 180.0) * cos(ellip.angle * CV_PI / 180.0) - (ellip.size.height / 2.0 - 2) * sin(i * CV_PI / 180.0) * sin(ellip.angle * CV_PI / 180.0);
		int y_in = ellip.center.y + (ellip.size.height / 2.0 - 2) * sin(i * CV_PI / 180.0) * cos(ellip.angle * CV_PI / 180.0) + (ellip.size.width / 2.0 - 2) * cos(i * CV_PI / 180.0) * sin(ellip.angle * CV_PI / 180.0);
		int x_out = ellip.center.x + (ellip.size.width / 2.0 + 2) * cos(i * CV_PI / 180.0) * cos(ellip.angle * CV_PI / 180.0) - (ellip.size.height / 2.0 + 2) * sin(i * CV_PI / 180.0) * sin(ellip.angle * CV_PI / 180.0);
		int y_out = ellip.center.y + (ellip.size.height / 2.0 + 2) * sin(i * CV_PI / 180.0) * cos(ellip.angle * CV_PI / 180.0) + (ellip.size.width / 2.0 + 2) * cos(i * CV_PI / 180.0) * sin(ellip.angle * CV_PI / 180.0);
		if (x_in<0 | x_in>img.cols - 1 | y_in<0 | y_in>img.rows - 1 | x_out<0 | x_out>img.cols - 1 | y_out<0 | y_out>img.rows - 1) continue;
		if (img.at<uchar>(y_out, x_out) < 230 & img.at<uchar>(y_in, x_in) < 230) {
			ellip_diff += (float)(img.at<uchar>(y_out, x_out)) - (float)(img.at<uchar>(y_in, x_in));
			++ellip_cnt;
		}
	}
	ellip_diff /= ellip_cnt;
	return ellip_diff;
}
RotatedRect EyeTracker::pupilReconstruction(vector<vector<Point>> edges, Mat img, Point coarsecenter) {
	RotatedRect pupil;
	double confidence = 0.0;
	if (edges.size() == 0) return pupil;
	Mat canny_starburst(img.rows, img.cols, CV_8UC1, Scalar(0));
	for (int i = 0; i < edges.size(); ++i) drawContours(canny_starburst, edges, i, Scalar(255));
	vector<Vec2i> pupiledge;
	vector<int> R;
	StarBurst(canny_starburst, coarsecenter, 100, pupiledge, R);
	RotatedRect pupil_star2 = pupilfitStarBurst(pupiledge, R, 100);
	if (pupil_star2.center.x > 0 & pupil_star2.center.y > 0) {
		confidence = EllipDifferiential(img, pupil_star2);
		pupil = pupil_star2;
	}
	if (edges.size() == 1) {
		RotatedRect pupil_tmp = fitEllipse(edges[0]);
		float ellip_diff = EllipDifferiential(img, pupil_tmp);
		if (ellip_diff > confidence) {
			pupil = pupil_tmp;
			confidence = ellip_diff;
		}
	}
	if (edges.size() == 2) {
		for (int i = 0; i < edges.size(); ++i) {
			RotatedRect pupil_tmp = fitEllipse(edges[i]);
			float ellip_diff = EllipDifferiential(img, pupil_tmp);
			if (ellip_diff > confidence) {
				pupil = pupil_tmp;
				confidence = ellip_diff;
			}
		}
		vector<Point> combine;
		combine.insert(combine.begin(), edges[0].begin(), edges[0].end());
		combine.insert(combine.begin(), edges[1].begin(), edges[1].end());
		RotatedRect pupil_tmp = fitEllipse(combine);
		float ellip_diff = EllipDifferiential(img, pupil_tmp);
		if (ellip_diff > confidence) {
			pupil = pupil_tmp;
			confidence = ellip_diff;
		}
	}
	if (edges.size() >= 3) {
		for (int i = 0; i < edges.size(); ++i) {
			RotatedRect pupil_tmp = fitEllipse(edges[i]);
			float ellip_diff = EllipDifferiential(img, pupil_tmp);
			if (ellip_diff > confidence) {
				pupil = pupil_tmp;
				confidence = ellip_diff;
			}
		}
		for (int i = 0; i < edges.size() - 1; ++i) {
			for (int j = i + 1; j < edges.size(); ++j) {
				vector<Point> combine;
				combine.insert(combine.begin(), edges[i].begin(), edges[i].end());
				combine.insert(combine.begin(), edges[j].begin(), edges[j].end());
				RotatedRect pupil_tmp = fitEllipse(combine);
				float ellip_diff = EllipDifferiential(img, pupil_tmp);
				if (ellip_diff > confidence) {
					pupil = pupil_tmp;
					confidence = ellip_diff;
				}
			}
		}
		for (int i = 0; i < edges.size() - 2; ++i) {
			for (int j = i + 1; j < edges.size() - 1; ++j) {
				for (int k = j + 1; k < edges.size(); ++k) {
					vector<Point> combine;
					combine.insert(combine.begin(), edges[i].begin(), edges[i].end());
					combine.insert(combine.begin(), edges[j].begin(), edges[j].end());
					combine.insert(combine.begin(), edges[k].begin(), edges[k].end());
					RotatedRect pupil_tmp = fitEllipse(combine);
					float ellip_diff = EllipDifferiential(img, pupil_tmp);
					if (ellip_diff > confidence) {
						pupil = pupil_tmp;
						confidence = ellip_diff;
					}
				}
			}
		}
	}
	return pupil;
}
RotatedRect EyeTracker::pupilDetect(Mat canny_img, Mat img, Point coarsecenter, int distance) {
	vector<vector<Point>> contours;
	edgeConnectivity(canny_img, contours);
	Mat edges;
	edges = Mat(canny_img.rows, canny_img.cols, CV_8UC1, Scalar(0));
	for (int i = 0; i < contours.size(); ++i) {
		drawContours(edges, contours, i, i + 1);
	}
	vector<int> edge_vote(contours.size(), 0);
	vector<Vec2i> Radial;
	vector<int> R_dis;
	StarBurst(canny_img, coarsecenter, distance, Radial, R_dis);
	for (int i = 0; i < Radial.size(); ++i) {
		if (R_dis[i] > 0) {
			if (edges.at<uchar>(Radial[i][1], Radial[i][0]) > 0 & edges.at<uchar>(Radial[i][1], Radial[i][0]) - 1 > -1 & edges.at<uchar>(Radial[i][1], Radial[i][0]) - 1 < edge_vote.size()) {
				edge_vote[edges.at<uchar>(Radial[i][1], Radial[i][0]) - 1]++;
			}
		}
	}
	vector<int> vote_idx(edge_vote.size());
	for (int i = 0; i < vote_idx.size(); ++i) vote_idx[i] = i;
	qsort(edge_vote, vote_idx, 0, edge_vote.size() - 1);
	int total_vote = 0;
	int vote_i = 0;
	vector<vector<Point>> reconstruction_edge;
	while (vote_i < vote_idx.size() && total_vote < 0.75 * Dnum) {
		if (edge_vote[vote_i] < 10) break;
		reconstruction_edge.push_back(contours[vote_idx[vote_i]]);
		total_vote += edge_vote[vote_i];
		vote_i++;
	}
	return pupilReconstruction(reconstruction_edge, img, coarsecenter);
}
void EyeTracker::findGlint(Mat img, vector<Point>& glints, int thresh, int kernel_size, Point pupil_center) {
	int ymin = pupil_center.y;
	if (ymin < 0) ymin = 0;
	int ymax = pupil_center.y + 150;
	if (ymax > img.rows - 1) ymax = img.rows - 1;
	int xmin = pupil_center.x - 150;
	if (xmin < 0) xmin = 0;
	int xmax = pupil_center.x + 150;
	if (xmax > img.cols - 1) xmax = img.cols - 1;
	if (xmax <= xmin || ymax <= ymin) return;
	Rect crop(Point(xmin, ymin), Point(xmax, ymax));
	Mat glint_img = img.clone();
	glint_img = glint_img(crop);
	threshold(glint_img, glint_img, thresh, 255, 0);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
	morphologyEx(glint_img, glint_img, MORPH_CLOSE, kernel);
	morphologyEx(glint_img, glint_img, MORPH_OPEN, kernel);
	threshold(glint_img, glint_img, 254, 255, 0);
	vector<vector<Point>> glint_contours;
	findContours(glint_img, glint_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	if (glint_contours.size() >= 2) {
		double min_dist = img.cols;
		vector<Point> glint_tmp(2, Point());
		for (int i = 0; i < glint_contours.size() - 1; ++i) {
			for (int j = i + 1; j < glint_contours.size(); ++j) {
				Rect r1 = boundingRect(glint_contours[i]);
				Rect r2 = boundingRect(glint_contours[j]);
				Point g1 = Point(r1.x, r1.y) + Point(r1.width / 2, r1.height / 2);
				Point g2 = Point(r2.x, r2.y) + Point(r2.width / 2, r2.height / 2);
				if (abs(g1.y - g2.y) < 15 && abs(g1.x - g2.x) > 30 && abs(g1.x - g2.x) < 100) {
					double dist = norm((g1 + g2) / 2 + Point(xmin, ymin) - pupil_center);
					if (dist < min_dist) {
						min_dist = dist;
						glint_tmp[0] = g1 + Point(xmin, ymin);
						glint_tmp[1] = g2 + Point(xmin, ymin);
					}
				}
			}
		}
		if (glint_tmp[0].x < glint_tmp[1].x) {
			glints.push_back(glint_tmp[0]);
			glints.push_back(glint_tmp[1]);
		}
		else {
			glints.push_back(glint_tmp[1]);
			glints.push_back(glint_tmp[0]);
		}
	}
}
void EyeTracker::GroundTruthDetect(Mat scene_) {
	groundtruth_pt = Point2f(NAN, NAN);
	Mat scene_gray;
	cvtColor(scene_, scene_gray, COLOR_BGR2GRAY);
	vector<Vec3f> circles;
	threshold(scene_gray, scene_gray, 200, 255, 0);
	pyrDown(scene_gray, scene_gray);
	HoughCircles(scene_gray, circles, HOUGH_GRADIENT, 1, 1000, 40, 15, 50, 100);
	vector<vector<Point>> centers;
	findContours(scene_gray, centers, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	pyrDown(scene_, scene_);
	for (int i = 0; i < circles.size(); ++i) {
		for (int j = 0; j < centers.size(); ++j) {
			Rect r = boundingRect(centers[j]);
			Point p = Point(r.x, r.y) + Point(r.width / 2, r.height / 2);
			if (norm(p - Point(circles[i][0], circles[i][1])) < 10) {
				circle(scene_, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(0, 0, 255), 3);
				circle(scene_, Point(circles[i][0], circles[i][1]), 5, Scalar(0, 0, 255), -1);
				circle(scene_, p, 5, Scalar(255, 0, 0), -1);
				groundtruth_pt = 2 * p;
				break;
			}
		}
	}
	imshow("scene", scene_);
	waitKey(1);
}
Point2f EyeTracker::FilterGaze(Point2f gazept) {
	gaze_seq.push_back(gazept);
	if (gaze_seq.size() > 5) gaze_seq.erase(gaze_seq.begin());
	vector<float> sorted_x, sorted_y;
	for (int i = 0; i < gaze_seq.size(); ++i) {
		sorted_x.push_back(gaze_seq[i].x);
		sorted_y.push_back(gaze_seq[i].y);
	}
	sort(sorted_x.begin(), sorted_x.end());
	sort(sorted_y.begin(), sorted_y.end());
	return Point2f(sorted_x[sorted_x.size() / 2], sorted_y[sorted_y.size() / 2]);
}
void EyeTracker::OnePointCompensation(Mat scene_, Point2f gaze_point) {
	Mat scene_gray;
	cvtColor(scene_, scene_gray, COLOR_BGR2GRAY);
	vector<Vec3f> circles;
	threshold(scene_gray, scene_gray, 200, 255, 0);
	pyrDown(scene_gray, scene_gray);
	HoughCircles(scene_gray, circles, HOUGH_GRADIENT, 1, 1000, 40, 15, 20, 100);
	vector<vector<Point>> centers;
	findContours(scene_gray, centers, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	pyrDown(scene_, scene_);
	for (int i = 0; i < circles.size(); ++i) {
		for (int j = 0; j < centers.size(); ++j) {
			Rect r = boundingRect(centers[j]);
			Point2f p = Point2f(r.x, r.y) + Point2f(r.width / 2, r.height / 2);
			if (norm(p - Point2f(circles[i][0], circles[i][1])) < 10) {
				circle(scene_, Point2f(circles[i][0], circles[i][1]), circles[i][2], Scalar(0, 0, 255), 3);
				circle(scene_, Point2f(circles[i][0], circles[i][1]), 5, Scalar(0, 0, 255), -1);
				circle(scene_, p, 5, Scalar(255, 0, 0), -1);
				onep_compensation = gaze_point - 2 * p;
				break;
			}
		}
	}
	cout << onep_compensation << endl;
	imshow("scene", scene_);
	waitKey();
}
void EyeTracker::arucoDetect(Mat scene_, std::vector<int>& ids, std::vector<std::vector<cv::Point2f>>& corners) {
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	cv::aruco::detectMarkers(scene_, dictionary, corners, ids);
}
void EyeTracker::do_process() {
	int fcnt = 0;
	if (!is_calib) {
		writer.open("gaze_result.avi", CV_FOURCC('X', 'V', 'I', 'D'), 20, Size(1920, 1080));
	}
	while (true) {
		eye1 >> eye1_img;
		eye2 >> eye2_img;
		eye3 >> eye3_img;
		eye4 >> eye4_img;
		scene >> scene_img;
		if (eye1_img.empty() | eye2_img.empty() | eye3_img.empty() | eye4_img.empty()) break;
		fcnt++;
		if (fcnt < 10) continue;/*0608*/
		if (is_calib) {
			if (fcnt < 0 || fcnt > 200) continue;
		}
		clock_t tim1 = clock();
		Mat eye1_gray, eye2_gray, eye3_gray, eye4_gray;
		cvtColor(eye1_img, eye1_gray, COLOR_BGR2GRAY);
		cvtColor(eye2_img, eye2_gray, COLOR_BGR2GRAY);
		cvtColor(eye3_img, eye3_gray, COLOR_BGR2GRAY);
		cvtColor(eye4_img, eye4_gray, COLOR_BGR2GRAY);
		coarse_center1 = findCoarseCenterFRST(eye1_gray);
		coarse_center2 = findCoarseCenterFRST(eye2_gray);
		coarse_center3 = findCoarseCenterFRST(eye3_gray);
		coarse_center4 = findCoarseCenterFRST(eye4_gray);
		circle(eye1_img, coarse_center1, 5, Scalar(0, 0, 255), -1);
		circle(eye2_img, coarse_center2, 5, Scalar(0, 0, 255), -1);
		circle(eye3_img, coarse_center3, 5, Scalar(0, 0, 255), -1);
		circle(eye4_img, coarse_center4, 5, Scalar(0, 0, 255), -1);
		Mat denoise1, denoise2, denoise3, denoise4;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
		morphologyEx(eye1_gray, denoise1, MORPH_OPEN, kernel);
		morphologyEx(eye2_gray, denoise2, MORPH_OPEN, kernel);
		morphologyEx(eye3_gray, denoise3, MORPH_OPEN, kernel);
		morphologyEx(eye4_gray, denoise4, MORPH_OPEN, kernel);
		Mat canny1, canny2, canny3, canny4;
		Canny(denoise1, canny1, 20, 40);
		Canny(denoise2, canny2, 20, 40);
		Canny(denoise3, canny3, 20, 40);
		Canny(denoise4, canny4, 20, 40);
		RotatedRect pupil1 = pupilDetect(canny1, eye1_gray, coarse_center1, 100);
		RotatedRect pupil2 = pupilDetect(canny2, eye2_gray, coarse_center2, 100);
		RotatedRect pupil3 = pupilDetect(canny3, eye3_gray, coarse_center3, 100);
		RotatedRect pupil4 = pupilDetect(canny4, eye4_gray, coarse_center4, 100);
		ellipse(eye1_img, pupil1, Scalar(255, 0, 0), 2);
		circle(eye1_img, pupil1.center, 8, Scalar(255, 0, 0), -1);
		ellipse(eye2_img, pupil2, Scalar(255, 0, 0), 2);
		circle(eye2_img, pupil2.center, 8, Scalar(255, 0, 0), -1);
		ellipse(eye3_img, pupil3, Scalar(255, 0, 0), 2);
		circle(eye3_img, pupil3.center, 8, Scalar(255, 0, 0), -1);
		ellipse(eye4_img, pupil4, Scalar(255, 0, 0), 2);
		circle(eye4_img, pupil4.center, 8, Scalar(255, 0, 0), -1);
		vector<Point> glints1, glints2, glints3, glints4;
		findGlint(eye1_gray, glints1, 240, 5, pupil1.center);
		findGlint(eye2_gray, glints2, 240, 5, pupil2.center);
		findGlint(eye3_gray, glints3, 240, 5, pupil3.center);
		findGlint(eye4_gray, glints4, 240, 5, pupil4.center);
		for (int i = 0; i < glints1.size(); ++i) {
			circle(eye1_img, glints1[i], 5, Scalar(0, 255, 0), -1);
			string text = to_string(i);
			putText(eye1_img, text, glints1[i], CV_FONT_BLACK, 1, Scalar(0, 0, 0));
		}
		for (int i = 0; i < glints2.size(); ++i) {
			circle(eye2_img, glints2[i], 5, Scalar(0, 255, 0), -1);
			string text = to_string(i);
			putText(eye2_img, text, glints2[i], CV_FONT_BLACK, 1, Scalar(0, 0, 0));
		}
		for (int i = 0; i < glints3.size(); ++i) {
			circle(eye3_img, glints3[i], 5, Scalar(0, 255, 0), -1);
			string text = to_string(i);
			putText(eye3_img, text, glints3[i], CV_FONT_BLACK, 1, Scalar(0, 0, 0));
		}
		for (int i = 0; i < glints4.size(); ++i) {
			circle(eye4_img, glints4[i], 5, Scalar(0, 255, 0), -1);
			string text = to_string(i);
			putText(eye4_img, text, glints4[i], CV_FONT_BLACK, 1, Scalar(0, 0, 0));
		}
		imshow("eye1", eye1_img);
		imshow("eye2", eye2_img);
		imshow("eye3", eye3_img);
		imshow("eye4", eye4_img);
		waitKey(1);

		if (is_calib) {
			GroundTruthDetect(scene_img);
			if (pupil1.center.x > 0 & pupil2.center.x > 0 & pupil3.center.x > 0 & pupil4.center.x > 0 &
				glints1.size() > 0 & glints2.size() > 0 & glints3.size() > 0 & glints4.size() > 0) {
				double datainarray[24] =
				{
					pupil1.center.x , pupil1.center.y ,glints1[0].x ,glints1[0].y, glints1[1].x , glints1[1].y ,
					pupil2.center.x , pupil2.center.y ,glints2[0].x ,glints2[0].y, glints2[1].x , glints2[1].y ,
					pupil3.center.x , pupil3.center.y ,glints3[0].x ,glints3[0].y, glints3[1].x , glints3[1].y ,
					pupil4.center.x , pupil4.center.y ,glints4[0].x ,glints4[0].y, glints4[1].x , glints4[1].y ,
				};
				Mat datain = Mat(24, 1, CV_64F, datainarray);
				datain = datain - xmin_glint;
				for (int i = 0; i < 24; i++)
					datain.at<double>(i, 0) = 2 * datain.at<double>(i, 0) / (xmaxarray_glint[i] - xminarray_glint[i] + 0.0) - 1;
				Mat N3 = W1_lin * datain + B1_lin;
				point_esti.x = 1 * ((N3.at<double>(0, 0) + 1) * (ymaxarray_glint[0] - yminarray_glint[0]) / 2.0 + yminarray_glint[0]);
				point_esti.y = 1 * ((N3.at<double>(1, 0) + 1) * (ymaxarray_glint[1] - yminarray_glint[1]) / 2.0 + yminarray_glint[1]);
				point_esti = FilterGaze(point_esti);
				cout << point_esti << endl;
				if (!isnan(groundtruth_pt.x) & !isnan(groundtruth_pt.y)) {
					calib_eye.push_back(Point3f(point_esti.x, point_esti.y, 1.0));
					calib_gaze.push_back(groundtruth_pt);
					double datainarray[26] =
					{
						pupil1.center.x , pupil1.center.y ,glints1[0].x ,glints1[0].y, glints1[1].x , glints1[1].y ,
						pupil2.center.x , pupil2.center.y ,glints2[0].x ,glints2[0].y, glints2[1].x , glints2[1].y ,
						pupil3.center.x , pupil3.center.y ,glints3[0].x ,glints3[0].y, glints3[1].x , glints3[1].y ,
						pupil4.center.x , pupil4.center.y ,glints4[0].x ,glints4[0].y, glints4[1].x , glints4[1].y ,
						groundtruth_pt.x, groundtruth_pt.y, 
					};
					if (fresult.is_open()) {
						for (int i = 0; i < 26; ++i) fresult << datainarray[i] << ',';
						fresult << endl;
					}
				}
			}
		}
		else {
			if (pupil1.center.x > 0 & pupil2.center.x > 0 & pupil3.center.x > 0 & pupil4.center.x > 0 &
				glints1.size() > 0 & glints2.size() > 0 & glints3.size() > 0 & glints4.size() > 0) {
				double eyedatainarray[25] =
				{
					1.0, pupil1.center.x , pupil1.center.y ,glints1[0].x ,glints1[0].y, glints1[1].x , glints1[1].y ,
					pupil2.center.x , pupil2.center.y ,glints2[0].x ,glints2[0].y, glints2[1].x , glints2[1].y ,
					pupil3.center.x , pupil3.center.y ,glints3[0].x ,glints3[0].y, glints3[1].x , glints3[1].y ,
					pupil4.center.x , pupil4.center.y ,glints4[0].x ,glints4[0].y, glints4[1].x , glints4[1].y ,
				};
				Mat eyedatain = Mat(25, 1, CV_64F, eyedatainarray);
				/* 0607 */
				double map[50] = {
					370.195965132843,- 7.07479827434610,0.190483809463987,4.96079422167127,- 1.89026952449369,6.05233735962377,3.21143052054153,- 7.65770633021852,0.610886219153407,6.14986910524570,5.02973667161782,- 7.01085975073632,- 3.06489273715714,- 2.78147737689085,2.11039797864249,1.03849954725863,- 6.38148924237123,3.83525260261164,- 0.623929117206155,- 0.842783172729246,- 1.61276119926693	,7.18381357078379,1.87412661001226,0.281657581505710,- 1.58283185848870 ,
					-1584.30804682340,- 0.160559813677915,4.46958037590252,3.13722389273976,- 1.51772072461967,- 5.00324889850259,- 0.685481809047196,- 2.25645109113388	,8.56983331259037,0.0926240448938590	,0.578383746115996,0.0870721319231178,- 0.788732734107445,0.551969489766254,4.40891838340579,- 0.832994140991754,- 0.946688893946976,2.26938034443060,- 1.46756602899105,- 0.197682403869252,0.233612252440951,5.14283607316582,- 0.468661557250081,3.75242441847539,- 7.49385134378021 ,
				};
				/* 0608 */
				/*double map[50] = {
					1133.70854153294,-8.04168132672028,2.67981550382351,9.58102923973563,-7.22136201137260,0.419780571838419,-0.750819060753065,-6.65494747937514,-0.763298348328903,4.92350268502098,-1.67132858287354,0.415588947572976,4.95752806800456,-2.53221836016944,-0.818475490844801,4.03219379168691,0.250983675731298,-0.280992829982163,-1.23945463966393,-6.13593052003893,0.190061573394832,3.50382403925188,0.235224231166724,1.14188408173639,3.42247632507872,
					908.315030367873,4.01381908428006,7.48940501209605,3.52490310058027,-1.29711022287821,-3.44078893630170,-5.25468292988540,3.22926885215585,4.71531915888799,-4.72471386510058,-3.88396335259548,-4.54855909904909,-0.0906942190427777,-3.35477993540964,2.59867864965715,0.318962841396286,-2.74892073764895,9.80675725480584,-1.65906429774240,-0.545232838991027,2.65751772645589,-1.47712702252455,1.39053816051824,-4.28259767385520,-0.766606457596952 
				};*/
				/* 0609 */
				/*double map[50] = {
					2082.74918274649,-1.36556934613450,-0.822669811014061,2.28976947753246,-0.182325259539202,4.08350298574147,3.23256274143085,-7.52724853515137,0.941245253359367,0.498491154111131,-1.12330139510777	,2.82826354479889,-3.71389324573411,-1.63056766648476,-1.41295603628292,-2.17202296512816,2.05711244195421,-1.85602463189116,1.53230134698505,-4.09954555290619,1.72553488320790,1.41208296984253,-1.45432011973857,4.08339668793974,-3.44436794301550,
					138.172779941450	,1.03681139812242,1.37273962029623,-4.27957668156096,-2.45058784192736,7.17801005806228,-0.942922425737338,-1.64161451131759,3.03218486216364	,1.15803201878787,5.62337496537533,-3.32454173197084,-2.00439340404106,1.17619082788850,6.13183496161210	,1.07682000164498,-3.27172272092657,0.444981335855229,-7.01284715499014,-3.52127102563973,3.74054167651151,-1.23162608823178,-2.42442494856834,5.26420704560247,-1.77221186139976,
				};*/
				/*double map[50] = {
				673.020200443016,- 0.827410587795361,- 0.0785196316452907,4.10154166125929,- 1.40461220662961,2.66104076981511,- 2.19776972340765,- 2.57835324321504,0.674282852162369,0.709674666356778,- 0.585602437946223,- 0.166534576343425,- 2.89515548167203,- 4.86977336394452,- 1.89524684481752,0.989201275050125,1.24499136853068,0.481221600273106,6.23014707422739,- 7.15003143050120,0.711514852964487,0.185639955583878,0.791128785043788,6.43253700436915,- 2.46346619307899,
					4940.66532633609,0.160886528816010,0.111083646860112,- 0.206451399903322,1.74187012775293,2.89799080041377,- 1.31563285619503,2.95601621637505,0.0245578704186176,- 5.76316333171366,- 7.82162490554615,- 1.47225238808283,- 3.08117232169356,- 3.48300716119660,8.20459171546991,- 0.750168821050775,- 0.778753483460175,2.75432698290998,- 4.66456286996838,0.221145762419774,4.60701315929194,5.51648060884302,- 5.04806819089775,- 9.06532081046752,2.82686880675006,
				};*/
				Mat map_M = Mat(2, 25, CV_64F, map);
				Mat gaze_personal_esti = map_M * eyedatain;
				Point2f point_personal_esti = Point2f(gaze_personal_esti.at<double>(0), gaze_personal_esti.at<double>(1));
				if (onep_compensation.x == 0 && onep_compensation.y == 0) {
					OnePointCompensation(scene_img, point_personal_esti);
				}
				point_personal_esti = point_personal_esti - onep_compensation;
				vector<int> ids;
				vector<vector<Point2f>> aruco_corners;
				arucoDetect(scene_img, ids, aruco_corners);
				if (ids.size() > 0) {
					cv::aruco::drawDetectedMarkers(scene_img, aruco_corners, ids);//绘制检测到的靶标的框
				}
				circle(scene_img, point_personal_esti, 20, Scalar(255, 0, 0), 8);
				circle(scene_img, point_personal_esti, 20, Scalar(255, 255, 0), 5);
				pyrDown(scene_img, scene_img);
				imshow("scene", scene_img);
				waitKey(1);
				if (fresult.is_open()) {
					fresult << fcnt << ',' << point_personal_esti.x << ',' << point_personal_esti.y << ',';
					for (int i = 1; i <= 10; ++i) {
						int j = 0;
						for (; j < ids.size(); ++j) {
							if (ids[j] == i) {
								fresult << aruco_corners[j][0].x << ',' << aruco_corners[j][0].y << ',';
								break;
							}
						}
						if(j== ids.size()) fresult << -1 << ',' << -1 << ',';
					}
					fresult << endl;
				}
			}
			if (scene_img.size().width < 1920) pyrUp(scene_img, scene_img);
			writer.write(scene_img);
		}
		cout << clock() - tim1 << "ms" << fcnt << endl;
	}

	eye1.release();
	eye2.release();
	eye3.release();
	eye4.release();
	scene.release();

}