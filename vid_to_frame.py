import cv2
import os
vidcap = cv2.VideoCapture(r"road_survey_vid\gps_cam1.mp4")
success,image = vidcap.read()
count = 0
path = r"C:\Users\manis\PycharmProjects\pothole_heatmap\frames"
while success:
  cv2.imwrite(os.path.join(path, "frame%d.jpg" % count), image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 100