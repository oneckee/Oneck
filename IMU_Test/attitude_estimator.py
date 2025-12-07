import math

class AttitudeEstimator:
    def estimate(self, ax, ay, az):
        # 가속도 기반 자세 추정 (단순 버전)
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        return roll, pitch
