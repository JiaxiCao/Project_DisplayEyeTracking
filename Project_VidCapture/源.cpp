#include<iostream> 
#include<opencv2/opencv.hpp>
#include "windows.h"
#include "dshow.h"


#define PATH "D:/"
#define W 720
#define H 576

using namespace std;
using namespace cv;

#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "quartz.lib")

int listDevices(vector<string>& list) {

	ICreateDevEnum* pDevEnum = NULL;
	IEnumMoniker* pEnum = NULL;
	int deviceCounter = 0;
	CoInitialize(NULL);

	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
		CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
		reinterpret_cast<void**>(&pDevEnum));


	if (SUCCEEDED(hr))
	{
		// Create an enumerator for the video capture category.
		hr = pDevEnum->CreateClassEnumerator(
			CLSID_VideoInputDeviceCategory,
			&pEnum, 0);

		if (hr == S_OK) {

			printf("SETUP: Looking For Capture Devices\n");
			IMoniker* pMoniker = NULL;

			while (pEnum->Next(1, &pMoniker, NULL) == S_OK) {

				IPropertyBag* pPropBag;
				hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
					(void**)(&pPropBag));

				if (FAILED(hr)) {
					pMoniker->Release();
					continue;  // Skip this one, maybe the next one will work.
				}


				// Find the description or friendly name.
				VARIANT varName;
				VariantInit(&varName);
				hr = pPropBag->Read(L"Description", &varName, 0);

				if (FAILED(hr)) hr = pPropBag->Read(L"FriendlyName", &varName, 0);

				if (SUCCEEDED(hr))
				{

					hr = pPropBag->Read(L"FriendlyName", &varName, 0);

					int count = 0;
					char tmp[255] = { 0 };
					//int maxLen = sizeof(deviceNames[0]) / sizeof(deviceNames[0][0]) - 2;
					while (varName.bstrVal[count] != 0x00 && count < 255)
					{
						tmp[count] = (char)varName.bstrVal[count];
						count++;
					}
					list.push_back(tmp);
					//deviceNames[deviceCounter][count] = 0;

					//if (!silent) DebugPrintOut("SETUP: %i) %s\n", deviceCounter, deviceNames[deviceCounter]);
				}

				pPropBag->Release();
				pPropBag = NULL;

				pMoniker->Release();
				pMoniker = NULL;

				deviceCounter++;
			}

			pDevEnum->Release();
			pDevEnum = NULL;

			pEnum->Release();
			pEnum = NULL;
		}

		//if (!silent) DebugPrintOut("SETUP: %i Device(s) found\n\n", deviceCounter);
	}

	//comUnInit();

	return deviceCounter;
}

int main(){

	vector<string> list;
	listDevices(list);
	int capid0 = 0, capid1 = 0;
	cout << "dev_size =      " << list.size() << endl;

	VideoCapture eye1, eye2, eye3, eye4, scene;
	int idx1 = 1, idx2 = 2, idx3 = 3, idx4 = 4, idx_scene = 0;
	for (int i = 0; i < list.size(); ++i) {
		cout << "camera index " << i << ": " << list[i] << endl;
		if (list[i] == "5M Cam") idx_scene = i;
	}

	bool camera_flag = false;
	while (!camera_flag) {
		eye1.open(idx1 + cv::CAP_DSHOW);
		eye2.open(idx2 + cv::CAP_DSHOW);
		eye3.open(idx3 + cv::CAP_DSHOW);
		eye4.open(idx4 + cv::CAP_DSHOW);
		if (!eye1.isOpened() || !eye2.isOpened() || !eye3.isOpened() || !eye4.isOpened())
		{
			std::cout << "Can not open camera.\n";
			return -1;
		}
		scene.open(idx_scene + cv::CAP_DSHOW);
		if (!scene.isOpened())
		{
			std::cout << "Can not open camera.\n";
			return -1;
		}

		eye1.set(cv::CAP_PROP_FRAME_WIDTH, W);
		eye1.set(cv::CAP_PROP_FRAME_HEIGHT, H);
		eye1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
		eye2.set(cv::CAP_PROP_FRAME_WIDTH, W);
		eye2.set(cv::CAP_PROP_FRAME_HEIGHT, H);
		eye2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
		eye3.set(cv::CAP_PROP_FRAME_WIDTH, W);
		eye3.set(cv::CAP_PROP_FRAME_HEIGHT, H);
		eye3.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
		eye4.set(cv::CAP_PROP_FRAME_WIDTH, W);
		eye4.set(cv::CAP_PROP_FRAME_HEIGHT, H);
		eye4.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
		scene.set(cv::CAP_PROP_AUTOFOCUS, 1);
		//scene.set(CAP_PROP_FOCUS, 5);
		scene.set(cv::CAP_PROP_FRAME_WIDTH, 1920);						/*Set frame size*/
		scene.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
		scene.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));	/*Default YUV2 codecs fps is 10, MJPEG codecs fps is 50+*/

		std::cout << "摄像机打开正确？ y/n" << endl;
		while (true) {
			Mat f1, f2, f3, f4, fs;
			eye1 >> f1;
			eye2 >> f2;
			eye3 >> f3;
			eye4 >> f4;
			if (f1.empty() | f2.empty() | f3.empty() | f4.empty()) break;
			scene >> fs;
			cv::pyrDown(fs, fs);
			cv::imshow("scene", fs);
			cv::pyrDown(f1, f1);
			cv::imshow("1", f1);
			cv::pyrDown(f2, f2);
			cv::imshow("2", f2);
			cv::pyrDown(f3, f3);
			cv::imshow("3", f3);
			cv::pyrDown(f4, f4);
			cv::imshow("4", f4);

			char kt = waitKey(1);
			if (kt == 27) {
				cvDestroyAllWindows();
				eye1.release();
				eye2.release();
				eye3.release();
				eye4.release();
				scene.release();
				return 0;
			}
			if (kt == 'y' || kt == 'Y') {
				camera_flag = true;
				break;
			}
			if (kt == 'n' || kt == 'N') {
				eye1.release();
				eye2.release();
				eye3.release();
				eye4.release();
				scene.release();
				idx1 = 0; idx2 = 1; idx3 = 2; idx4 = 3; idx_scene = 6;
				eye1.open(idx1 + cv::CAP_DSHOW);
				eye2.open(idx2 + cv::CAP_DSHOW);
				eye3.open(idx3 + cv::CAP_DSHOW);
				eye4.open(idx4 + cv::CAP_DSHOW);
				if (!eye1.isOpened() || !eye2.isOpened() || !eye3.isOpened() || !eye4.isOpened())
				{
					std::cout << "Can not open camera.\n";
					return -1;
				}
				scene.open(idx_scene + cv::CAP_DSHOW);
				if (!scene.isOpened())
				{
					std::cout << "Can not open camera.\n";
					return -1;
				}
				eye1.set(cv::CAP_PROP_FRAME_WIDTH, W);
				eye1.set(cv::CAP_PROP_FRAME_HEIGHT, H);
				eye1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
				eye2.set(cv::CAP_PROP_FRAME_WIDTH, W);
				eye2.set(cv::CAP_PROP_FRAME_HEIGHT, H);
				eye2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
				eye3.set(cv::CAP_PROP_FRAME_WIDTH, W);
				eye3.set(cv::CAP_PROP_FRAME_HEIGHT, H);
				eye3.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
				eye4.set(cv::CAP_PROP_FRAME_WIDTH, W);
				eye4.set(cv::CAP_PROP_FRAME_HEIGHT, H);
				eye4.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
				scene.set(cv::CAP_PROP_AUTOFOCUS, 1);
				//scene.set(CAP_PROP_FOCUS, 5);
				scene.set(cv::CAP_PROP_FRAME_WIDTH, 1920);						/*Set frame size*/
				scene.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
				scene.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));	/*Default YUV2 codecs fps is 10, MJPEG codecs fps is 50+*/

			}
		}
	}
	cvDestroyAllWindows();
	string uname;
	std::cout << "文件保存路径：" << (string)PATH;
	std::cin >> uname;
	string path = (string)PATH + uname + "/";
	std::cout << "路径：" << path << endl;
	
	string out1 = path + "eye1.avi";
	string out2 = path + "eye2.avi";
	string out3 = path + "eye3.avi";
	string out4 = path + "eye4.avi";
	string out5 = path + "scene.avi";

	VideoWriter writer1(out1, CV_FOURCC('X', 'V', 'I', 'D'), 20, Size(W, H), true);
	VideoWriter writer2(out2, CV_FOURCC('X', 'V', 'I', 'D'), 20, Size(W, H), true);
	VideoWriter writer3(out3, CV_FOURCC('X', 'V', 'I', 'D'), 20, Size(W, H), true);
	VideoWriter writer4(out4, CV_FOURCC('X', 'V', 'I', 'D'), 20, Size(W, H), true);
	VideoWriter writer5(out5, CV_FOURCC('X', 'V', 'I', 'D'), 20, Size(1920, 1080), true);

	std::cout << "Press SPACE to start recording" << std::endl;
	while (true)
	{
		Mat f1, f2, f3, f4, fs;
		eye1 >> f1;
		eye2 >> f2;
		eye3 >> f3;
		eye4 >> f4;
		if (f1.empty() | f2.empty() | f3.empty() | f4.empty()) break;
		scene >> fs;
		pyrDown(fs, fs);
		imshow("scene", fs);
		pyrDown(f1, f1);
		imshow("1", f1);
		pyrDown(f2, f2);
		imshow("2", f2);
		pyrDown(f3, f3);
		imshow("3", f3);
		pyrDown(f4, f4);
		imshow("4", f4);

		char kt = waitKey(1);
		if (kt == 32) break;
	}
	cout << "\a";
	cout << "Start Capturing ..." << endl;

	int framecnt = 0;
	while (true)
	{
		Mat f1, f2, f3, f4, fs;
		eye1 >> f1;
		eye2 >> f2;
		eye3 >> f3;
		eye4 >> f4;
		if (f1.empty() | f2.empty() | f3.empty() | f4.empty()) break;
		++framecnt;
		scene >> fs;

		writer1.write(f1);
		writer2.write(f2);
		writer3.write(f3);
		writer4.write(f4);
		writer5.write(fs);
		pyrDown(fs, fs);
		imshow("scene", fs);
		pyrDown(f1, f1);
		imshow("1", f1);
		pyrDown(f2, f2);
		imshow("2", f2);
		pyrDown(f3, f3);
		imshow("3", f3);
		pyrDown(f4, f4);
		imshow("4", f4);


		char k = waitKey(1);
		if (k == 27) break;
	}
	cvDestroyAllWindows();
	writer1.release();
	writer2.release();
	writer3.release();
	writer4.release();
	writer5.release();

	return 0;
}

