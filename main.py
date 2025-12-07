import time
import random
import math

dt = 0.001  # 50Hz

# PID gains
Kp = 0.8
Ki = 0.0
Kd = 0.15

TARGET = {"roll": 0.0, "pitch": 0.0, "yaw": 45.0}

# PID 저장 변수
integral_roll = 0
integral_pitch = 0
integral_yaw = 0
prev_roll_err = 0
prev_pitch_err = 0
prev_yaw_err = 0

# 실제 기체 상태
true_roll = 0.0
true_pitch = 0.0
true_yaw = 0.0

# 센서 측정용 상태
imu_roll = 0.0
imu_pitch = 0.0
imu_yaw = 0.0


def wrap_angle(x):
    return (x + 180) % 360 - 180


def read_imu(true_r, true_p, true_y):
    # 센서 노이즈 추가된 값
    noisy_r = true_r + random.gauss(0, 0.2)
    noisy_p = true_p + random.gauss(0, 0.2)
    noisy_y = true_y + random.gauss(0, 0.2)

    return noisy_r, noisy_p, noisy_y


print("==== 시작 ====")

while True:
    # ---- 1) IMU 측정 ----
    imu_roll, imu_pitch, imu_yaw = read_imu(true_roll, true_pitch, true_yaw)

    # ---- 2) PID 오차 ----
    re = TARGET["roll"] - imu_roll
    pe = TARGET["pitch"] - imu_pitch
    ye = wrap_angle(TARGET["yaw"] - imu_yaw)

    # ---- 3) 적분 ----
    integral_roll += re * dt
    integral_pitch += pe * dt
    integral_yaw += ye * dt

    # ---- 4) 미분 ----
    derr_roll = (re - prev_roll_err) / dt
    derr_pitch = (pe - prev_pitch_err) / dt
    derr_yaw = (ye - prev_yaw_err) / dt

    prev_roll_err = re
    prev_pitch_err = pe
    prev_yaw_err = ye

    # ---- 5) PID 출력 ----
    roll_cmd = Kp * re + Ki * integral_roll + Kd * derr_roll
    pitch_cmd = Kp * pe + Ki * integral_pitch + Kd * derr_pitch
    yaw_cmd = Kp * ye + Ki * integral_yaw + Kd * derr_yaw

    # ---- 6) Dynamics: PID 명령이 실제 기체를 움직임 ----
    # (여기서 0.1은 '기체 민감도(게인)'임 — 조종 명령 얼마나 잘 따라가는지)
    true_roll += roll_cmd * 0.1 * dt
    true_pitch += pitch_cmd * 0.1 * dt
    true_yaw += yaw_cmd * 0.1 * dt *5

    true_yaw = wrap_angle(true_yaw)

    # ---- 7) 출력 ----
    print(f"현재 YAW = {imu_yaw:.2f}°,  목표까지 남은 오차 = {ye:.2f}°,   CMD = {yaw_cmd:.2f}")

    time.sleep(dt)
