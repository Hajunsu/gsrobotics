import cv2
from utilities.gelsightmini import GelSightMini

imgw = 640
imgh = 480
brd_frac = 0.05 # default 0.15

def show_image():
    counter = 0
    cam = GelSightMini(
        target_width=imgw,
        target_height=imgh,
        border_fraction=brd_frac)

    deviceidx = cam.select_device(device_idx=1)

    cam.start()
    while True:
        img = cam.update(1.0)
        imbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame_rgb', imbgr)

        if cv2.waitKey(1) == ord('q'):
            print("end")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':

    show_image()
