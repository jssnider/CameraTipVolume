import genericpath
import sys
import time
from TipMeasurement import CameraInterface

DEFAULT_CAPTURE_TIME = 2
DEFAULT_VIDEO_FILENAME = "foo.avi"
DEFAULT_LOG_FILENAME = "log.csv"

def capture(duration=DEFAULT_CAPTURE_TIME, vid_filename=DEFAULT_VIDEO_FILENAME, log_filename=DEFAULT_LOG_FILENAME):
    print(f"capture to {vid_filename} for {duration} seconds")
    video_writer = CameraInterface(vid_filename, tipName = "P50", src=1)
    video_writer.start_recording()
    video_writer.start_measuring()
    video_writer.show_message("Measuring fluid")
    time.sleep(duration)
    video_writer.stop_measuring()
    v = video_writer.get_volume()
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    PrintAndLog(f"{t},{v}", file=log_filename, header="Timestamp,Volume")
    video_writer.stop_recording()

def PrintAndLog(s: str, file = DEFAULT_LOG_FILENAME, header = ''):
    print(s)
    FileAlreadyExists = genericpath.exists(file)
    fp = open(file, 'a')
    if header != '' and not FileAlreadyExists:
        header = f'{header}\n'
        fp.writelines(header)
    line = f'{s}\n'
    fp.writelines(line)
    fp.close()    

##############################################################################
if __name__ == '__main__':
    duration = DEFAULT_CAPTURE_TIME # assume default time
    vfn = DEFAULT_VIDEO_FILENAME
    lfn = DEFAULT_LOG_FILENAME
    if len(sys.argv) >= 2:
        print(f"{len(sys.argv)=}")
        for arg in sys.argv:
            if str.isnumeric(arg): # just a number must be duration
                duration = float(arg)
            elif arg.startswith("file:"): # base of filenames
                    print(f"file {arg=}")
                    param = arg[len("file:"):]
                    print(f"file {param=}")
                    vfn = param + ".avi"
                    lfn = param + ".csv"
    capture(duration, vfn, lfn)
