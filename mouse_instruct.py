import serial

class MouseInstruct:
    def __init__(self, port="COM6", baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"[INFO] Connected to Arduino on {port}")
        except Exception as e:
            print(f"[ERROR] Could not open serial port {port}: {e}")
            self.ser = None

    @staticmethod
    def get_mouse(port="COM6"):
        return MouseInstruct(port)

    def move(self, dx, dy):
        if not self.ser: 
            return
        # Clamp values between -128 and 127
        dx = max(-128, min(127, int(dx)))
        dy = max(-128, min(127, int(dy)))
        try:
            # Send as signed bytes
            self.ser.write(dx.to_bytes(1, byteorder='little', signed=True) +
                           dy.to_bytes(1, byteorder='little', signed=True))
        except Exception as e:
            print(f"[ERROR] Serial write: {e}")

    def silent_flick(self, dx, dy):
        # Same as move, can be modified for special Arduino behavior
        self.move(dx, dy)

    def close(self):
        if self.ser:
            self.ser.close()
            print("[INFO] Serial port closed")
