#Step 1.
#Take a video frame placed in \road_survey_vid\upload_folder. it strips the video into frames
import cv2
import os
Root_dir = os.path.abspath(".")


upload_folder_dir = os.path.join(Root_dir, r"road_survey_vid/upload_folder" )

vid_list= os.listdir(upload_folder_dir)
print(vid_list)
upload_vid_loc = os.path.join(upload_folder_dir, vid_list[0])
print(upload_vid_loc)
vidcap = cv2.VideoCapture(upload_vid_loc)
success,image = vidcap.read()
count = 0
path = os.path.join(Root_dir, r"frames")
# code below will extract one frame after every 0.1 sec, but will change the frequency of picking frame laatter when tuning
#seconds = 100
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print(fps)

frame_no = 0
while success:
  frame_no += 1
  cv2.imwrite(os.path.join(path, "{}_frame{}.jpg".format(vid_list[0],count)), image)     # save frame as JPEG file
  print("{}_frame{}.jpg".format(vid_list[0],count))
  success,image = vidcap.read()
  count += 1 # this stes the frequency of capturing frames
  print('Read a new frame at ', count, " :", success)

