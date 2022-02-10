#include"EyeTracking.h"

#define PATH "D:/0401-cjx/ahmu/0607/C/"
#define TPATH "D:/0401-cjx/ahmu/0607/U/"
int main() {
	EyeTracker eye;
	//eye.Init_Camera((string)PATH + "eye1.avi", (string)PATH + "eye2.avi", (string)PATH + "eye3.avi", (string)PATH + "eye4.avi", (string)PATH + "scene.avi", true, (string)PATH + "result.csv");
	eye.Init_Camera((string)TPATH + "eye1.avi", (string)TPATH + "eye2.avi", (string)TPATH + "eye3.avi", (string)TPATH + "eye4.avi", (string)TPATH + "scene.avi", false, (string)TPATH + "gaze_result.csv");
	eye.Init(); 
	eye.do_process();
	return 0;
}