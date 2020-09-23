import cv2
import os
vid_list= os.listdir(r"road_survey_vid\upload_folder")
vidcap = cv2.VideoCapture(r"road_survey_vid\upload_folder\%s"%vid_list[0])
success,image = vidcap.read()
count = 0
path = r"C:\Users\manis\PycharmProjects\pothole_heatmap\frames"
# code below will extract one frame after every 0.1 sec, but will change the frequency of picking frame laatter when tuning
#seconds = 100
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print(fps)
multiplier = int(fps * seconds)
frame_no = 0
while success:
  frame_no += 1
  cv2.imwrite(os.path.join(path, "{}_frame{}.jpg".format(vid_list[0],count)), image)     # save frame as JPEG file
  print("{}_frame{}.jpg".format(vid_list[0],count))
  success,image = vidcap.read()
  count += 1 # this stes the frequency of capturing frames
  print('Read a new frame at ', count, " :", success)

