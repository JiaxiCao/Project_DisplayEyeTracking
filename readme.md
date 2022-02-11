## Dependencies
opencv 3.4.8 + opencv_contrib 3.4.8

## Project_ArucoDetect
* Aruco detection test, aruco and dictionary should match

## Project_GazeTrackingFeature
* extract eye gaze feature from calibration data, and gaze tracking test
* Calibration

    * circle marker is used in calibration, change the marker detection code in `GroundTruthDetect()` if calibration marker has been changed

    * pupil and glint features are saved, file name is an input of `Init_Camera()`

    * only use the data collected after user concentrated on marker, estimate valid frame range and change it in `do_process()`

    * use `matlab` to fit personal parameters (TODO: C++ realization of personal calibration)

* Validation

    * one point calibration used before validation, make sure a one point calibration is done in data collection. if not, remove `OnePointCompensation()` in `do_process()`

    * gaze point and aruco position in scene image are saved, file name is an input of `Init_Camera()`

    * raw gaze tracking in HMD scene video is saved in `gaze_result.avi`