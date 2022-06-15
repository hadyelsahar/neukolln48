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
LAST_CAMERA_FRAME = None
LAST_CAPTURED_OBJECT = None
DELAY_CAMERA= 0
DELAY_OBJECT_DETECTION = 0
DELAY_VIDEO_PLAYING=0
BREAK = False
ZONES = {"OUT-1":(0,0.6),"IN-1":(0.6,0.7),"IN-2":(0.7,0.8),"IN-3":(0.8,0.9),"OUT-2":(0.9,1)}
######################
# OBJECT RECOGNITION #
######################

def camera_capture(video_path=None):
    global LAST_CAMERA_FRAME, DELAY_CAMERA, BREAK

    if video_path is not None:
        vid = cv2.VideoCapture(video_path)
        print("INFO::Reading from VIDEO {video_path}")
        # DELAY_CAMERA = 0.1
    else:
        vid = cv2.VideoCapture(int(args.camera))
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
    global LAST_CAMERA_FRAME, DELAY_OBJECT_DETECTION, LAST_CAPTURED_OBJECT, BREAK
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # define a video capture object
    while not BREAK:
        if LAST_CAMERA_FRAME is not None:
            results = model(LAST_CAMERA_FRAME)
            # Results
            df = results.pandas().xyxy[0]
            LAST_CAPTURED_OBJECT = df

        time.sleep(DELAY_OBJECT_DETECTION)


####################
# ZONES WORK LOGIC #
####################

def annotate_locations(df, frame_shape, zones):
    max_y, max_x, _ = frame_shape
    print(frame_shape)
    locations = []
    for y in df.ymax:
        y = y / max_y
        l = [z_name for z_name, z in ZONES.items() if  z[0] <= y <= z[1]][0]
        locations.append(l)

    df["locations"] = locations
    return df

def which_zone(df):
    """function takes a dataframe and decides how to resolve which video to play

    Args:
        df (pandas.dataframe): pandas dataframe that contains capture objects
    Returns: 0,1,2,3 integer, 0 random , 1 left, 2 center and 3 right
    """

    if df[df.name=="person"].empty: # no person in frame
        return 0

    else:
        df = df[df.name=="person"].copy()
        df = annotate_locations(df, LAST_CAMERA_FRAME.shape, ZONES)
        print(df)
        df["IN"] = df.apply(lambda x: x.locations.split("-")[0] == "IN", axis=1)

        ## todo: select the person closest
        # in case of multiple people select the person closest to the center
        ######

        if len(df[df["IN"]]) > 0:
            location = df[df["IN"]].locations.values[0].split("-")
            return int(location[1])

        else:
            return 0 # all persons are in the OUT zone

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

#######################################
# VIDEO DISPLAYING 2 videos at a time #
#######################################


def display_video():
    global LAST_CAMERA_FRAME, DELAY_VIDEO_PLAYING, LAST_CAPTURED_OBJECT, BREAK

    videos = {
        0: {
            "l": cv2.VideoCapture('./videos/Channel_0_Left.mp4'),
            "r": cv2.VideoCapture('./videos/Channel_0_Right.mp4'),
        },
        1: {
            "l": cv2.VideoCapture('./videos/Channel_1_Left.mp4'),
            "r": cv2.VideoCapture('./videos/Channel_1_Right.mp4'),
        },
        2: {
            "l": cv2.VideoCapture('./videos/Channel_3_Left.mp4'),
            "r": cv2.VideoCapture('./videos/Channel_3_Right.mp4'),
        },
        3: {
            "l": cv2.VideoCapture('./videos/Channel_2_Left.mp4'),
            "r": cv2.VideoCapture('./videos/Channel_2_Right.mp4'),
        }
    }

    # all videos are correctly loaded
    for _, v in videos.items():
        assert v["r"].isOpened() and v["l"].isOpened(), "video couldn't load"

    out_monitor = "camera"
    out_r = "video_r"
    out_l = "video_l"

    cv2.namedWindow(out_monitor, cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow(out_r, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(out_r, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow(out_l, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(out_l, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while not BREAK:

        if LAST_CAMERA_FRAME is None or LAST_CAPTURED_OBJECT is None:
            continue

        df = LAST_CAPTURED_OBJECT.copy()
        video_r = videos[which_zone(df)]["r"]
        video_l = videos[which_zone(df)]["l"]
        ret_r, video_frame_r = video_r.read()
        ret_l, video_frame_l = video_l.read()


        LAST_CAMERA_FRAME = highlight_zones(LAST_CAMERA_FRAME, ZONES)

        if LAST_CAMERA_FRAME is not None:
            cv2.imshow(out_monitor, LAST_CAMERA_FRAME)

        if ret_r and ret_l:
            cv2.imshow(out_r, video_frame_r)
            cv2.imshow(out_l, video_frame_l)
        else:
            print('no video')
            video_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(DELAY_VIDEO_PLAYING)

    video_r.release()
    video_l.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', metavar="v", default=None)
    parser.add_argument('--camera', metavar="c", default=2, type=int)

    args = parser.parse_args()
    t1 = threading.Thread(target=camera_capture, args=(args.video,))
    t2 = threading.Thread(target=object_detection)
    t1.start()
    t2.start()
    display_video()

