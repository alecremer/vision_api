import sys
import cv2

run_mode = sys.argv[1]

def rtsp_client(ip):

    stream = cv2.VideoCapture("rtsp://" + ip)
    while True:

        ret, frame = stream.read()

        if not ret:
            break

        cv2.imshow('Stream', frame)
        key = cv2.waitKey(1)
        if key == 27: # esc 
            break

    stream.release()
    cv2.destroyAllWindows()


