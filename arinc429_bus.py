class ARINC429Bus:
    def __init__(self):
        # TX/RX 초기화
        pass

    def send_attitude_command(self, roll_cmd, pitch_cmd, yaw_cmd):
        frame = self.encode_attitude_frame(roll_cmd, pitch_cmd, yaw_cmd)
        self.tx(frame)

    def encode_attitude_frame(self, roll, pitch, yaw):
        # ARINC429 포맷에 맞게 데이터 구성
        return {
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw
        }

    def tx(self, frame):
        print("[ARINC429 TX]", frame)

    def read_sensors(self):
        # 센서 데이터를 ARINC429로 RX받을 때
        return {
            "accel": [0, 0, -9.8],
            "gyro": [0.01, 0.02, -0.01]
        }


"""class ARINC429Bus:
    def __init__(self):
        self.buffer = None

    def send(self, data):
        # 실제 ARINC429는 32비트 워드로 보내지만 단순화
        self.buffer = data

    def receive(self):
        return self.buffer
"""