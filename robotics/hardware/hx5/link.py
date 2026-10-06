"""UDP link host -> container bridge (bridge.py). Packet = seq (uint32) + 20 joint targets (float64, rad, sim/hand.py order).

Standard library only: bridge.py imports this inside the container.
"""
import socket
import struct

HOST, PORT = "127.0.0.1", 9870
PACKET = struct.Struct("<I20d")


class HandLink:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr, self.seq = (host, port), 0

    def send(self, q):
        self.seq += 1
        self.sock.sendto(PACKET.pack(self.seq, *map(float, q)), self.addr)

    def close(self):
        self.sock.close()
