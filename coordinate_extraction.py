# Step 2: Run coordinate_extraction.py
# crop the frames into 3 images 1) Longitude, 2)latitude 3)Required portion of image
# latitude and longitude images are proceses in pytessract to extract the coordinates.
# Required portion of the image is saved in the folder \road_images for further pot hole detection
# a csv file is created which stores all, latitude and longitude value extractcted by pytessract against image name
# this csv file will be populted further in next step.
import cv2
import os, pandas as pd
from PIL import Image
import pytesseract
import shutil


vid_list= os.listdir(r"road_survey_vid\upload_folder")
vidcap = cv2.VideoCapture(r"road_survey_vid\upload_folder\%s"%vid_list[0])
success,image = vidcap.read()
vid_fps = vidcap.get(cv2.CAP_PROP_FPS)
print(vid_fps)
extract_fps = 59
multiplier = int(vid_fps/extract_fps)
print(multiplier)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_coordinate(frame):
    frame_loc = "frames\%s"%frame
    #print(frame_loc)
    image = Image.open(frame_loc)
    img_crop = image.crop((93, 56, 626, 1006))
    img_crop = img_crop.rotate(90, Image.NEAREST, expand=1)
    img_crop.save(r"road_images\%s"%frame)
    lat_crop = image.crop((306, 1221, 565, 1269))
    long_crop = image.crop((336, 1308, 601, 1359))
    lat_crop.save("lat.jpg")
    long_crop.save("long.jpg")
    long= cv2.imread("long.jpg")
    lat= cv2.imread("lat.jpg")
    try:
        latitude = float(pytesseract.image_to_string(lat))
    except ValueError:
        latitude = 0.00000
        src = os.path.join(r"C:\Users\manis\PycharmProjects\pothole_heatmap\frames",frame)
        shutil.copy(src, r"C:\Users\manis\PycharmProjects\pothole_heatmap\wrong_coordinate_extacrtion")

    try:
        longitude = float(pytesseract.image_to_string(long))
    except ValueError:
        longitude = 0.00000
        src = os.path.join(r"C:\Users\manis\PycharmProjects\pothole_heatmap\frames", frame)
        shutil.copy(src, r"C:\Users\manis\PycharmProjects\pothole_heatmap\wrong_coordinate_extacrtion")

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
        print(count, frame)
        lat, long = extract_coordinate(frame)
        row_entry = [frame, lat, long]
        core_df.loc[len(core_df)] = row_entry
        core_df.to_csv("core_data.csv")
    count = count + 1
print(core_df)










