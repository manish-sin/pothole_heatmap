# import cv2
# import os
# vid_list= os.listdir(r"C:\Users\manis\PycharmProjects\pothole_heatmap\holes")
# loc = r"C:\Users\manis\PycharmProjects\pothole_heatmap\holes_resize"
# i=1
# for frame in vid_list:
#     print(i)
#     i=i+1
#     img = cv2.imread(r"C:\Users\manis\PycharmProjects\pothole_heatmap\holes\%s"%frame)
#     img = cv2.resize(img, (3680,2760))
#     os.chdir(loc)
#     cv2.imwrite(frame, img)
import pandas as pd
import numpy as np
df= pd.read_csv("core_data.csv")
print(df)
lis = ["dsx", "54", "sfr", "gdf"]
my_array = np.array(lis)
df = df.set_index("Image")
#df["list"] = []
arr = np.array( [[ 1, 2, 3],[ 4, 2, 5]] )

#df.loc['gps_cams1.mp4_frame1038.jpg','Latitude'] = arr

print(df)

#df["RoIS"] = lis
