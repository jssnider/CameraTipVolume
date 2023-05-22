import socket
import sys

PORT = 65432

def send_string(host, port, message, wait:bool=True):
    BUFFER_SIZE = 1024
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Sender is sending on {port=}.")

    try:
        sock.connect((host, port))
        sock.sendall(message.encode())
        if wait:
            sock.settimeout(1)
            response = sock.recv(BUFFER_SIZE)
            print(f"{response=}")
    finally:
        sock.close()

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        send_string("localhost", PORT, ' '.join(sys.argv[1:]), False)
    else:
        send_string("localhost", PORT, "test")

