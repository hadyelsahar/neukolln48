import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# define a video capture object
vid = cv2.VideoCapture(0)
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    results = model(frame)
    # Results
    df = results.pandas().xyxy[0]
    FRAME_HAS_PERSON = not df[df.name=="person"].empty
    print(f"camera frame {FRAME_HAS_PERSON}")

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()