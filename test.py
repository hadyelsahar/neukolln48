import cv2
import threading


q = list()
def process():
    cap = cv2.VideoCapture("./videos/1.mp4")
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()
        q.append(frame)

def Display():
  while True:
    if len(q) > 0:
        frame = q.pop(0)
        cv2.imshow("frame1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == '__main__':
#  start threads
    p1 = threading.Thread(target=process)
    p1.start()
    Display()