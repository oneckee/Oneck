import random
import time
import math

class IMUSensor:

    def __init__(self):
        self.last_time = time.time()

        # --- 실제 센서처럼 보이도록 추가한 부분 ----
        self.bias_gx = random.uniform(-0.05, 0.05)  # 자이로 바이어스
        self.bias_gy = random.uniform(-0.05, 0.05)
        self.bias_gz = random.uniform(-0.05, 0.05)

        self.gyro_rw_gx = 0.0    # random walk 용 변수
        self.gyro_rw_gy = 0.0
        self.gyro_rw_gz = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def _gyro_random_walk(self, state):
        # 실제 센서처럼 시간 지날수록 drift 증가
        state += random.uniform(-0.01, 0.01)
        return state

    def read_raw_gyro(self):
        # Random Walk 업데이트
        self.gyro_rw_gx = self._gyro_random_walk(self.gyro_rw_gx)
        self.gyro_rw_gy = self._gyro_random_walk(self.gyro_rw_gy)
        self.gyro_rw_gz = self._gyro_random_walk(self.gyro_rw_gz)

        # 기본 회전 + 바이어스 + random walk + 노이즈
        gx = self.bias_gx + self.gyro_rw_gx + random.gauss(0, 0.01)
        gy = self.bias_gy + self.gyro_rw_gy + random.gauss(0, 0.01)
        gz = self.bias_gz + self.gyro_rw_gz + random.gauss(0, 0.01)

        return gx, gy, gz

    def read_raw_accel(self):

        # 중력 가속도 + 약간 기울어진 상태 + 노이즈 추가
        ax = random.gauss(0.0, 0.05)
        ay = random.gauss(0.0, 0.05)
        az = 9.81 + random.gauss(0.0, 0.05)

        return ax, ay, az

    def read_attitude(self):

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        ax, ay, az = self.read_raw_accel()
        gx, gy, gz = self.read_raw_gyro()

        # 가속도 센서 기준 Roll/pitch 추정
        acc_roll = math.degrees(math.atan2(ay, az))
        acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))

        # 자이로 적분
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt 

        # 보정 (Complementary Filter)
        alpha = 0.98
        self.roll = alpha * self.roll + (1 - alpha) * acc_roll
        self.pitch = alpha * self.pitch + (1 - alpha) * acc_pitch
        self.yaw += gz * dt
        
            
        
            

        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw
        }










"""
import random
import math



class IMUSensor:
    def __init__(self):
        pass

    def read(self):
        # 가속도(ax, ay, az), 각속도(gx, gy, gz)
        ax = random.uniform(-0.2, 0.2)
        ay = random.uniform(-0.2, 0.2)
        az = 9.8 + random.uniform(-0.3, 0.3)

        gx = random.uniform(-0.02, 0.02)
        gy = random.uniform(-0.02, 0.02)
        gz = random.uniform(-0.02, 0.02)

        return ax, ay, az, gx, gy, gz
"""