import cv2
import torch

import asyncio
import threading
import time



FRAME_HAS_PERSON = False
LAST_CAMERA_FRAME = None
DELAY_CAMERA= 0.001
DELAY_OBJECT_DETECTION = 0.001
DELAY_VIDEO_PLAYING=0.001
######################
# OBJECT RECOGNITION #
######################

def camera_capture():
    global LAST_CAMERA_FRAME, DELAY_CAMERA
    vid = cv2.VideoCapture(0)
    while True:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # Display the resulting frame
        if ret:
            LAST_CAMERA_FRAME = frame
            time.sleep(DELAY_CAMERA)

def object_detection():
    global FRAME_HAS_PERSON, LAST_CAMERA_FRAME, DELAY_OBJECT_DETECTION
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # define a video capture object
    while True:
        if LAST_CAMERA_FRAME is not None:
            results = model(LAST_CAMERA_FRAME)
            # Results
            df = results.pandas().xyxy[0]
            FRAME_HAS_PERSON = not df[df.name=="person"].empty
            print(f"camera frame has person: {FRAME_HAS_PERSON}")
            time.sleep(DELAY_OBJECT_DETECTION)

####################
# VIDEO DISPLAYING #
####################

def display_video():
    global FRAME_HAS_PERSON, LAST_CAMERA_FRAME, DELAY_VIDEO_PLAYING
    video1 = cv2.VideoCapture('./videos/1.mp4')
    video2 = cv2.VideoCapture('./videos/2.mp4')


    while(video1.isOpened() and video2.isOpened()):
        print(f"VIDEO FRAME while {FRAME_HAS_PERSON}")

        cap = video1 if FRAME_HAS_PERSON else video2
        ret, video_frame = cap.read()

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

# async def display_video():
#     global FRAME_HAS_PERSON

#     # video1 = cv2.VideoCapture('./videos/1.mp4')
#     # video2 = cv2.VideoCapture('./videos/2.mp4')

#     # while True:
#     #     display_video = video1 if FRAME_HAS_PERSON else video2
#     #     ret, video_frame = display_video.read()
#     #     # cv2.imshow("video", video_frame)

#     #     if cv2.waitKey(25) & 0xFF == ord('q'):
#     #         break

#     print(f"video frame {FRAME_HAS_PERSON}")
#     await asyncio.sleep(0)

# asyncio.gather(object_detection(), display_video())
# asyncio.get_event_loop().run_forever()

t1 = threading.Thread(target=camera_capture)
t2 = threading.Thread(target=object_detection)
t1.start()
t2.start()
# t2 = threading.Thread(target=display_video).start()
display_video()

