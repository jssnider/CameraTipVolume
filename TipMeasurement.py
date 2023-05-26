import matplotlib.pyplot as plt
from threading import Thread, Lock
from multiprocessing import Pool
from multiprocessing import Process, Queue, Pipe, Semaphore
import numpy as np
import imutils
from imutils import perspective
import cv2 # pip install opencv-python
import ctypes
import datetime
# import pause
import sys
import pandas as pd
from scipy import interpolate
import os
import time

# I'm leaving the hooks in for reading and displaying pressure sensor data.  This can be re-enabled when we
# get the new boards with pressure sensors on them.

class CameraInterface(object):
    def __init__(self, video_file_name, tipName = None, src=0): # src is for which camera to use
        # Create a VideoCapture object
        self.PIXELS_PER_MM = 582 / 51.36 # majik number to convert pixel to height for lookup to get a volume
        self.frame_name = video_file_name  # if using webcams, else just use src as it is.
        self.video_file = video_file_name
        self.img_counter = 0
        self.frame_count = 0
        self.frameX = 320
        self.frameY = 240
        self.measuring_active = False
        self.fluid_visual_height = 0
        self.fluid_visual_sum = 0
        self.fluid_visual_count = 0

        # We have pregenerated lookup tables to convert height to volume.
        if tipName == "P50":
            filestring = "lookup_tables\989330-P50-internal volume-barrier38.53mm-mandrel43.61mm.SLDPRT-VolumeTable.csv"
            data = pd.read_csv(filepath_or_buffer=filestring, skiprows=[0], usecols=[0, 1],
                               encoding='iso-8859-1')  # dataframe type
        else: #add more tip types
            filestring = "lookup_tables\Microtip Total Volume - B21690.csv"
            data = pd.read_csv(filepath_or_buffer=filestring, skiprows=[0], usecols=[0, 1],
                               encoding='iso-8859-1')  # dataframe type
        data.columns = (data.columns.str.strip().str.replace(u'µ', 'u'))
        h = data.loc[:, ['Height (mm)']].values.astype('float64')
        v = data.loc[:, ['Volume (uL)']].values.astype('float64')

        h = h.reshape(-1, )  # I don't know why but it seems to read in as a vector
        v = v.reshape(-1, )
        # Now initialize the interpolate class so we can do lookups later.
        self.f_h = interpolate.interp1d(h, v, kind="linear")  # linear interpolation
        self.persistent_volume = -1.0
        self.img_grab = None
        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        # self.codec = cv2.VideoWriter_fourcc(*'H264')
        # self.codec = cv2.VideoWriter_fourcc(*'X264')

        # "return a timestamp in microseconds (us)"
        self.start_us = -1
        self.frame_us = -1

        # to be able to view frame by frame it is better to avoid better compression codecs
        # they derive frames based on previous frames making it hard to go backwards.
        # Set up codec and output video settings
        # self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        # self.codec = cv2.VideoWriter_fourcc(*'H264')
        # self.codec = cv2.VideoWriter_fourcc(*'X264')
        # self.codec = cv2.VideoWriter_fourcc(*'MJPG')
        # self.codec = cv2.VideoWriter_fourcc(*'H264')
        # The codec and file extension need to be compatible
        if self.video_file != "":
            self.video_file_name = self.video_file #+ '.avi'
        self.pressure = 0
        self.frame_counter = 0
        self.message = "Init"
        self.already_written_message = ""
        self.level_sense_height = 0
        # initialize the video camera stream and read the first frame
        # from the stream
        # CAP_ANY
        # CAP_DSHOW
        # CAP_FFMPEG
        # CAP_MSMF
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'UYVY'))
        # self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_SHARPNESS, 1)  # 0 to 127, default 1
        self.capture.set(cv2.CAP_PROP_GAMMA, 218)  # 40 to 500, default 218
        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # -15 to 15, default 0
        self.capture.set(cv2.CAP_PROP_CONTRAST, 25)  # 0 to 30, default 15
        self.capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4700)  # 1000 to 10000, default auto
        self.capture.set(cv2.CAP_PROP_GAIN, 0)  # 0 to 100. Increases noise and speed
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2) #       ### Set width as desired ###
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #    ### Set height as desired ###
        self.capture.set(cv2.CAP_PROP_FPS, 35.0)
        self.capture.set(cv2.CAP_PROP_ZOOM, 100)  # 100 to 800, default 100
        self.capture.set(cv2.CAP_PROP_PAN, 0)  # -180 to 180, default 0
        self.capture.set(cv2.CAP_PROP_TILT, 0)  # -180 to 180, default 0
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)  # 1.0 for auto
        
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, -5)  # 2^-6 = 1/64 s the bare minimum for 60 fps

        (self.status, self.frame) = self.capture.read()
        self.newFrame = None

        self._is_running = True
        self.recording_thread = None

        # store the image dimensions, initialize the video writer,
        # and construct the zeros array
        # Default resolutions of the frame are obtained (system dependent)
        # self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        if self.video_file != "":
            # print("Creating video file with " + repr(fps) + " fps at " + repr(int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))) + "x"
            #       + repr(int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) + " resolution.")
            w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Creating video file with {fps} fps at {w} x {h} resolution.")
            # Rotated -> self.output_video = cv2.VideoWriter(self.video_file_name, self.codec, fps, (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            self.output_video = cv2.VideoWriter(self.video_file_name, self.codec, fps, (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Start the thread to read frames from the video stream
        self._lock = Lock()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        # Start another thread to show/save frames
        if self.video_file != "":
            print('initialized {}'.format(self.video_file))

    def update(self):
        # Read the next frame from the stream in a different thread
        while self._is_running:
            if self.capture.isOpened():
                (self.status, frametemp) = self.capture.read()
                # Using cv2.rotate() method -> self.frame = cv2.rotate(frametemp.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.frame = frametemp.copy()
                if self.video_file != "" :
                    tics = ctypes.c_int64()
                    freq = ctypes.c_int64()
                    # get ticks on the internal ~2MHz QPC clock
                    ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
                    # get the actual freq. of the internal ~2MHz QPC clock
                    ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
                    t_us = tics.value * 1e6 / freq.value
                    if self.start_us == -1:
                        self.start_us = t_us
                    self.frame_us = t_us

    def show_frame(self):
        # Display frames in main program
        if self.status:
            #newFrame = cv2.flip(self.frame, 0)
            with self._lock:
                newFrame = self.frame.copy()
            self.img_grab = newFrame.copy()
            if self.measuring_active and (self.frame_counter > 2):
                newFrame = self.show_measure_frame(newFrame)

            # display = format(self.pressure, '6.5f')
            # cv2.putText(newFrame, "Pressure " + display + " inches H2O", (10, 30),
            #             cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
            if self.message != "":
                cv2.putText(newFrame, self.message, (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
            localtime = datetime.datetime.now().isoformat()
            # '2021-07-14T13:01:51.840000'
            cv2.putText(newFrame, localtime, (10, 610), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
            display = "Frame %d" % self.frame_counter
            cv2.putText(newFrame, display, (10, 570), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)
            self.frame_counter = self.frame_counter + 1
            cv2.imshow(self.frame_name, newFrame)
            if self.video_file != "":
                self.output_video.write(newFrame)
        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            if self.video_file != "":
                self.output_video.release()
            self._is_running = False
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('p'):
            if self.img_grab is not None:
                localtimestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                img_name = "images\\Framegrab_{}.png".format(localtimestring)
                cv2.imwrite(img_name, self.img_grab)
                print("{} written!".format(img_name))

    def show_measure_frame(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        blur = cv2.medianBlur(gray, 5)
        # cv2.imshow("blur", blur)
        X_edged = cv2.Canny(blur, 30, 80)
        # cv2.imshow("X_edged", X_edged)
        # Display frames in main program
        # horizontal = X_edged.copy()
        vertical = X_edged.copy()
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        # cv2.imshow("vertical", vertical)
        # X_erode = cv2.erode(vertical, vertical_structure)
        # cv2.imshow("X_erode", X_erode)
        kernel = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(X_erode, cv2.MORPH_OPEN, kernel)
        X_dilate = cv2.dilate(vertical, horizontal_structure, iterations=4)
        # X_dilate = cv2.dilate(vertical, None, iterations=2)
        # cv2.imshow("X_dilate", X_dilate)
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        broken_line_v = np.array([[0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 0]], dtype=np.uint8)
        close = cv2.morphologyEx(X_dilate, cv2.MORPH_CLOSE, broken_line_v, iterations=3)
        # cv2.imshow("close", close)
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # create list object
        measure = []
        orig = image.copy()
        origWidth = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        origHeight = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # contour_image = cv2.cvtColor(close  , cv2.COLOR_GRAY2BGR)
        for c in cnts:
            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order.
            box = perspective.order_points(box)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            ((tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)) = box
            # each of these is a y, x coordinate.  We want a tall and narrow object.

            # cv2.drawContours(orig, [c.astype("int")], -1, (0, 0, 255), 1)
            # if not (tr_x - tl_x < 100 and tr_x - tl_x > 20 and br_y - tr_y > 250
            #     and tr_x < 600 and tl_x > 40 and bl_y > 440 and tl_y >= 0):
            #     print(box)
            # if bl_y < 478:
            #     bl = bl + (0, 2)
            #     bl_y = int(bl_y + 2)
            # if br_y < 478:
            #     br = br + (0, 2)
            #     br_y = int(br_y + 2)
            BBTopWidth = tr_x - tl_x 
            BBRightHeight = br_y - tr_y
            BBTopWidthMaxPercent = 100/480
            BBTopWidthMinPercent = 20/480
            BBRightHeightMinPercent = 250/640
            BBTopRightMaxPercent = 550/640
            BBTopLeftMinPercent = 40/640
            BBBottomLeftMaxPercent = 600/640

            contour = orig.copy()
            cv2.drawContours(contour, [c.astype("int")], -1, (0, 0, 255), 1)
            # cv2.imshow("contour", contour)
            # cv2.waitKey(0)

            # 640x480 majik numbers --> if ( BBTopWidth < 100 and BBTopWidth > 20 and BBRightHeight > 250 and tr_x < 550 and tl_x > 40 and bl_y < 600 and tl_y >= 0):
            if (    BBTopWidth < BBTopWidthMaxPercent * origWidth 
                and BBTopWidth > BBTopWidthMinPercent * origWidth 
                and BBRightHeight > BBRightHeightMinPercent * origHeight
                and tr_x < BBTopRightMaxPercent * origWidth 
                and tl_x > BBTopLeftMinPercent * origWidth 
                and bl_y < BBBottomLeftMaxPercent * origWidth 
                and tl_y >= 0
                ):
                    #
                    # if (tr_x - tl_x < 100 and tr_x - tl_x > 20 and br_y - tr_y > 250):
                #print ("found it: " + repr(box))

                # draw contours in red
                cv2.drawContours(orig, [c.astype("int")], -1, (0, 0, 255), 1)

                # Now draw our blue box
                cv2.line(orig, tuple(tl.astype("int")), tuple(tr.astype("int")), (255, 0, 0), 1)
                cv2.line(orig, tuple(tr.astype("int")), tuple(br.astype("int")), (255, 0, 0), 1)
                cv2.line(orig, tuple(br.astype("int")), tuple(bl.astype("int")), (255, 0, 0), 1)
                cv2.line(orig, tuple(bl.astype("int")), tuple(tl.astype("int")), (255, 0, 0), 1)

                # We have a bounded box and within that box we have an outline of the pipette
                # If we go to the same spot every time we should be able to detect clinging fluid
                # We are going to search inside the bounds to find the height of the green fluid.
                # More positive moves y down and positive moves x right
                finalY = br_y
                lostColor = 0
                # for y in range(min(self.frame_height - 1, int(br_y)), int(tr_y), -1):
                for y in range(min(orig.shape[0] - 1, int(br_y)), int(tr_y), -1):
                    InsideColor = False
                    StartedRed = False
                    #sys.stdout.write(str(y) + ": ")
                    if lostColor > 5:
                        break
                    lostColor = lostColor + 1
                    # for x in range(int(tl_x), min(self.frame_width - 1,  int(tr_x)), 1):
                    for x in range(int(tl_x), min(orig.shape[1] - 1,  int(tr_x)), 1):
                        if InsideColor:
                            b, g, r = orig[y, x]
                            # run until we hit red contour again
                            # sys.stdout.write("(" + str(b) + "," + str(g) + "," + str(r) + ") \r\n")
                            if b == 0 and g == 0 and r == 255:
                                break
                            else:
                                # if r < 255 and g < 255 and b < 255 and (g * 10 + b * 4) // 14 - r > 20:
                                if int(r+1)/int(b+1) > 1.2 : # 'liquid check'
                                    finalY = y
                                    lostColor = 0
                                    #orig[y,x] = (0,0,0)
                                    #contour_image[y,x] = (0,0,0)
                        else:
                            # color order is BGR
                            b, g, r = orig[y, x]
                            if b == 0 and g == 0 and r == 255:
                                # the next pixel after this is our measuring range
                                InsideColor = True
                    # print "."
                cv2.line(orig, (int(tl_x), int(finalY)), (int(tr_x), int(finalY)), (255, 255, 0), 1)
                # pixels per inch = xxx pixels * height measured" =
                self.fluid_visual_height = (br_y - finalY) # was previously divided by 134 for some reason
                self.fluid_visual_sum = self.fluid_visual_sum + self.fluid_visual_height
                self.fluid_visual_count = self.fluid_visual_count + 1
                self.fluid_visual_height = self.fluid_visual_sum / self.fluid_visual_count
                self.frame_count = self.frame_count + 1
                if self.frame_count == 30:
                    self.frame_count = 0
                    h, w, c = orig.shape
                    max_x = h - 100 # We need enough room to print
                    max_y = w - 30
                    self.frameX = int(min(max_x, br_x + 20))
                    self.frameY = int(min(max_y, br_y + 20))
                    # draw the object area on the image
                    self.persistent_volume = float(self.f_h(self.fluid_visual_height / self.PIXELS_PER_MM))
            if self.persistent_volume >= 0.0:
                txt = f"{self.fluid_visual_height:0.2f} pixels --> {self.persistent_volume:0.3f} uL"
                cv2.putText(orig, txt, (self.frameX-200, self.frameY), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.65, (0, 32, 255), 2)
        return orig

    # def save_frame(self):
    #     # Save obtained frame into video output file
    #     newFrame = cv2.flip(self.frame, 0)
    #     self.output_video.write(newFrame)

    def stop_recording(self):
        print("stopping recording")
        self._is_running = False
        self.thread.join()
        self.recording_thread.join()
        self.capture.release()
        if self.video_file != "":
            self.output_video.release()
        del(self.capture)
        # destroyAllWindows also cleans up the camera resource
        # cv2.destroyAllWindows()

    def start_recording_thread(self):
        while self._is_running:
            try:
                self.show_frame()
                # self.save_frame()
            except AttributeError:
                pass

    def start_recording(self):
        # Create another thread to show/save frames
        # start making measurements
        print("starting recording")
        self._is_running = True
        self.recording_thread = Thread(target=self.start_recording_thread, args=())
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def start_measuring(self):
        print("beginning measuring")
        self.measuring_active = True
        self.frame_count = 0
        self.fluid_visual_sum = 0
        self.fluid_visual_count = 0
        self.fluid_visual_height = 0

    def stop_measuring(self):
        print("ending measuring")
        self.measuring_active = False

    def check_running(self):
        return self._is_running

    def show_message(self, newMessage):
        self.message = newMessage

    # def get_pressure(self):
    #     return self.pressure

    def set_level(self, level):
        self.level_sense_height = level

    def get_height(self):
        if self.fluid_visual_count > 0:
            self.fluid_visual_height = self.fluid_visual_sum / self.fluid_visual_count
        else:
            self.fluid_visual_height = -1.0
        return self.fluid_visual_height

    def get_volume(self):
        self.persistent_volume = -1.0
        if self.fluid_visual_count > 0:
            self.fluid_visual_height = self.fluid_visual_sum / self.fluid_visual_count
            self.persistent_volume = self.f_h(self.fluid_visual_height / self.PIXELS_PER_MM)
        return float(self.persistent_volume)


##############################################################################
if __name__ == '__main__': # TESTING
    DEFAULT_CAPTURE_TIME = 2
    filename = "baz.avi"

    video_writer = CameraInterface(filename, tipName = "P50", src=1)
    video_writer.start_recording()
    video_writer.start_measuring()
    video_writer.show_message("Measuring fluid")
    time.sleep(DEFAULT_CAPTURE_TIME)
    video_writer.stop_measuring()
    v = video_writer.get_volume()
    print(f"{v=}")
    video_writer.stop_recording()
    exit(0)
