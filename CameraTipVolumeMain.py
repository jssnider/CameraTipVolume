import genericpath
import sys
import time
from TipMeasurement import CameraInterface
import socket
import inspect

PORT = 65432
CAMERA_NUMBER = 1
DEFAULT_CAPTURE_TIME = 2
DEFAULT_VIDEO_FILENAME = "foo.avi"
DEFAULT_LOG_FILENAME = "log.csv"

def capture(duration:str=DEFAULT_CAPTURE_TIME, vid_filename:str=DEFAULT_VIDEO_FILENAME, log_filename:str=DEFAULT_LOG_FILENAME):
    print(f"capture to {vid_filename} for {duration} seconds")
    video_writer = CameraInterface(vid_filename, tipName = "P50", src=CAMERA_NUMBER)
    video_writer.start_recording()
    video_writer.start_measuring()
    video_writer.show_message("Measuring fluid")
    time.sleep(int(duration))
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

def receive(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        sock.bind(("localhost", port))
        sock.listen(1)
        print(f"Receiver is listening on {port=}.")
        
        while True:
            conn, addr = sock.accept()
            data = conn.recv(1024).decode()
            print(f"Received: {data=}")
            response = f"Received {data=}"
            conn.sendall(response.encode())
            conn.close()

            call ="call "
            if data.lower().__contains__("stop"):
                break
            elif data.lower().startswith(call):
                s = data.removeprefix(call).split(" ")
                func = globals().get(s[0])
                if callable(func):
                    sig = inspect.signature(func)
                    sig_params = sig.parameters
                    params = s[1:]
                    if len(sig_params) == len(params):
                        print(f"calling {s}")
                        func(*params)
    finally:
        sock.close()

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
    receive(PORT) # Use this to wait for command to come in
    # capture(duration, vfn, lfn) # Use this to record for a given duration

