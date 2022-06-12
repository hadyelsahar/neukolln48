import cv2
import torch

import asyncio
import threading
import time
import argparse
import pyshine as ps
import numpy as np
#########################
####  GLOBAL PARAMS #####
#########################
FRAME_HAS_PERSON = False
LAST_CAMERA_FRAME = None
LAST_CAPTURED_OBJECT = None
DELAY_CAMERA= 0.001
DELAY_OBJECT_DETECTION = 0.001
DELAY_VIDEO_PLAYING=0.1
BREAK = False
ZONES = {"OUT ZONE 1":(0,0.6),"IN ZONE 1":(0.6,0.7),"IN ZONE 2":(0.7,0.8),"IN ZONE 3":(0.8,0.9),"OUT ZONE 2":(0.9,1)}
######################
# OBJECT RECOGNITION #
######################

def camera_capture(video_path=None):
    global LAST_CAMERA_FRAME, DELAY_CAMERA, BREAK

    if video_path is not None:
        vid = cv2.VideoCapture(video_path)
        print("INFO::Reading from VIDEO {video_path}")
        DELAY_CAMERA = 0.1
    else:
        vid = cv2.VideoCapture(1)
        print("INFO::Reading from CAMERA")

    while not BREAK:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # Display the resulting frame
        if ret:
            LAST_CAMERA_FRAME = frame
            time.sleep(DELAY_CAMERA)

def object_detection():
    global FRAME_HAS_PERSON, LAST_CAMERA_FRAME, DELAY_OBJECT_DETECTION, LAST_CAPTURED_OBJECT, BREAK
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # define a video capture object
    while not BREAK:
        if LAST_CAMERA_FRAME is not None:
            results = model(LAST_CAMERA_FRAME)
            # Results
            df = results.pandas().xyxy[0]
            LAST_CAPTURED_OBJECT = df
            FRAME_HAS_PERSON = not df[df.name=="person"].empty

            # print(f"camera frame has person: {FRAME_HAS_PERSON}")
            time.sleep(DELAY_OBJECT_DETECTION)


#################
# ZONES WORK ####
#################
def annotate_locations(df, frame_shape, zones):
    max_y, max_x, _ = frame_shape
    print(frame_shape)
    locations = []
    for y in df.ymax:
        y = y / max_y
        l = [z_name for z_name, z in ZONES.items() if  z[0] <= y <= z[1]]
        print(l)
        locations.append(l)

    df["locations"] = locations
    return df

def highlight_zones(frame, zones):
    y_max, x_max, _ = frame.shape
    for z_name, z in zones.items():
        x_offset = 50
        y_offset = round( ((z[0] + z[1]) / 2) * y_max)
        frame = ps.putBText(frame,z_name,
                            text_offset_x=x_offset,
                            text_offset_y=y_offset,
                            vspace=0,hspace=10,
                            font_scale=0.5,
                            background_RGB=(228,225,222),
                            text_RGB=(12,12,12))


        # Initialize blank mask image of same dimensions for drawing the shapes
        shapes = np.zeros_like(frame, np.uint8)
        # Draw shapes
        cv2.rectangle(shapes, (0, round(z[0]*y_max)), (x_max, round(z[1]*y_max)), (255, 255, 255), cv2.FILLED)

        # Generate output by blending image with shapes image, using the shapes
        # images also as mask to limit the blending to those parts

        out = frame.copy()
        alpha = 0.9
        mask = shapes.astype(bool)

        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        frame = out
    return frame


####################
# VIDEO DISPLAYING #
####################
def display_video():
    global FRAME_HAS_PERSON, LAST_CAMERA_FRAME, DELAY_VIDEO_PLAYING, LAST_CAPTURED_OBJECT, BREAK
    video1 = cv2.VideoCapture('./videos/1.mp4')
    video2 = cv2.VideoCapture('./videos/2.mp4')


    while(video1.isOpened() and video2.isOpened() and not BREAK):
        # print(f"VIDEO FRAME while {FRAME_HAS_PERSON}")
        if LAST_CAMERA_FRAME is None:
            continue

        cap = video1 if FRAME_HAS_PERSON else video2
        ret, video_frame = cap.read()
        if FRAME_HAS_PERSON:
            df = LAST_CAPTURED_OBJECT
            df = df[df.name=="person"]
            df = annotate_locations(df, LAST_CAMERA_FRAME.shape, ZONES)
            print(df)

        LAST_CAMERA_FRAME = highlight_zones(LAST_CAMERA_FRAME, ZONES)
        if LAST_CAMERA_FRAME is not None:
            cv2.imshow("camera", LAST_CAMERA_FRAME)
        if ret:
            cv2.imshow("video", video_frame)
        else:
            print('no video')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(DELAY_VIDEO_PLAYING)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', metavar="v", default=None)
    args = parser.parse_args()
    t1 = threading.Thread(target=camera_capture, args=(args.video,))
    t2 = threading.Thread(target=object_detection)
    t1.start()
    t2.start()
    display_video()

