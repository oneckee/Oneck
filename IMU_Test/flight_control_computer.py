class FlightControlComputer:
    def compute_control(self, roll, pitch):
        # 단순한 제어법칙 예시
        aileron = -roll * 10    # Roll 안정화
        elevator = -pitch * 8   # Pitch 안정화
        return aileron, elevator
