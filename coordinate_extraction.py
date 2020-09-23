import cv2
import os, pandas as pd
from PIL import Image
import pytesseract

vid_list= os.listdir(r"road_survey_vid\upload_folder")
vidcap = cv2.VideoCapture(r"road_survey_vid\upload_folder\%s"%vid_list[0])
success,image = vidcap.read()
vid_fps = vidcap.get(cv2.CAP_PROP_FPS)
print(vid_fps)
extract_fps = 5
multiplier = int(vid_fps/extract_fps)
print(multiplier)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_coordinate(frame):
    frame_loc = "frames\%s"%frame
    #print(frame_loc)
    image = Image.open(frame_loc)
    img_crop = image.crop((93, 117, 626, 1010))
    img_crop.save(r"road_images\%s"%frame)
    lat_crop = image.crop((312, 1247, 581, 1294))
    long_crop = image.crop((346, 1333, 587, 1385))
    lat_crop.save("lat.jpg")
    long_crop.save("long.jpg")
    long= cv2.imread("long.jpg")
    lat= cv2.imread("lat.jpg")
    latitude = float(pytesseract.image_to_string(lat))
    longitude = float(pytesseract.image_to_string(long))
    #cv2.imshow("longitude", long)
    #cv2.imshow("latitude", lat)
    cv2.waitKey(0)
    return latitude, longitude
frames = os.listdir("frames")
columns = ["Image", "Latitude", "Longitude"]
core_df = pd.DataFrame(columns=columns)
count = 0
for frame in frames:
    if count % multiplier == 0:
        print(count)
        lat, long = extract_coordinate(frame)
        row_entry = [frame, lat, long]
        core_df.loc[len(core_df)] = row_entry
        core_df.to_csv("core_data.csv")
    count = count + 1
print(core_df)










