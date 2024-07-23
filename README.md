**Tennis Ball Pathing**

This project aims to track a yellow tennis ball in a video stream and fit a parabolic trajectory to its movement. The script uses OpenCV lib for image processing and computer visionn, including color filtering, circle detection, and trajectory analysis.

**Overview**

* Video Capture: Reads video from a file or webcam.
* Yellow Ball Detection: Filters and detects yellow balls in the video.
* Circle Detection: Identifies circles corresponding to the detected balls.
* Trajectory Fitting: Computes and visualizes the parabolic trajectory of the ball's movement based on detected positions.
* Real-time Visualization: Displays the processed video with overlays of detected circles, tracking points, and the fitted trajectory.

**Dependencies**
* OpenCV
* NumPy

**Running the Project**

1. Setup Video Source: The script is currently set up to read from a file (TennisBall/vid4Edit_Trim.mp4). To use a webcam, uncomment the appropriate line in the code.

2. Compile and run the Main class.