##########################################################################################################
#라이브러리 불러오고 드론 연결
##########################################################################################################
import cv2
import threading
import time
from dronekit import connect, VehicleMode

# 드론에 연결하기 위한 연결 문자열
connection_string = 'udp:127.0.0.1:14550'

# 드론에 연결
vehicle = connect(connection_string, wait_ready=True)

# 연결이 완료될 때까지 기다림
while not vehicle.is_armable:
    print("Waiting for vehicle to initialize...")
    time.sleep(1)

print("Vehicle is now armable.")

# 드론의 아랫면 카메라에 대한 비디오 캡처 설정
cap_bottom = cv2.VideoCapture(1)  # 아랫면 카메라에 대한 비디오 캡처 설정
cap_side = cv2.VideoCapture(0)  # 측면 카메라에 대한 비디오 캡처 설정

#############################################################################################################
#PID제어로 사람과 드론 속도 맞추기
##############################################################################################################
# PID 제어 상수
Kp = 0.1  # 비례 상수
Ki = 0.01  # 적분 상수
Kd = 0.01  # 미분 상수

# 이전 오차 값들을 저장하기 위한 변수들
prev_error = 0
sum_error = 0

# 드론 속도 제어 함수
def control_drone_speed(distance_error):
    global prev_error, sum_error

    # PID 제어 알고리즘 계산
    error = distance_error
    delta_error = error - prev_error
    sum_error += error

    # PID 제어 값 계산
    pid_output = Kp * error + Ki * sum_error + Kd * delta_error

    # 드론 속도를 제어값에 따라 조절
    drone_speed = pid_output

    # 이전 오차 값 갱신
    prev_error = error

    return drone_speed

# 사람 인식 함수
def detect_people():
    while True:
        ret, frame = cap_side.read()  # 프레임 읽기
        if not ret:
            break

        # Haar Cascade를 이용하여 사람 인식
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 사람이 인식되었을 때
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # 사람의 몸 중심 계산
                person_center_x = x + w // 2
                person_center_y = y + h // 2

                # TODO: 드론과 사람 사이의 거리 계산
                # 사람의 몸 중심과 드론의 중심 간의 거리를 픽셀 기준으로 계산
                # distance_error = ...

                # PID 제어를 이용하여 드론 속도 조절
                drone_speed = control_drone_speed(distance_error)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_side.release()
    cv2.destroyAllWindows()

    #####################################################################################################
    #레일 따라 드론이 비행하도록 하기
    ######################################################################################################
# 드론 비행 경로 계획
def plan_flight_path(lane_info):
    # TODO: 인식된 차선 정보를 이용하여 비행 경로 계획
    # 예를 들어, 차선 중앙을 따라 비행하도록 경로를 계획

    # 비행 경로를 계획하여 드론에게 전달
    flight_path = []  # 예시로 빈 리스트 반환, 실제 경로 계획 필요
    return flight_path

# 드론 비행 제어
def control_drone_flight(flight_path):
    # 드론 비행 제어 로직
    for waypoint in flight_path:
        # 드론에게 waypoint로 이동하도록 명령
        # TODO: Roll, Pitch, Yaw를 조절하여 비행 경로를 따라갈 수 있도록 제어

        # 각 waypoint에서 일정 시간 동안 대기
        time.sleep(1)  # 1초 대기 (실제 시나리오에 맞게 조절)

if __name__ == "__main__":
    # 사람 인식과 드론 제어를 멀티스레딩으로 동시에 실행
    thread1 = threading.Thread(target=detect_people)
    thread1.start()

    while True:
        ret, frame = cap_bottom.read()  # 프레임 읽기
        if not ret:
            break

        # TODO: frame을 이용하여 차선 인식 코드 작성
        # 인식된 차선 정보를 lane_info 변수에 저장

        # 비행 경로 계획
        flight_path = plan_flight_path(lane_info)

        # TODO: 드론과 사람 사이의 거리 계산
        distance_error = 0  # 예시로 0으로 초기화, 실제로 거리 계산 필요

        # PID 제어를 이용하여 드론 속도 조절
        drone_speed = control_drone_speed(distance_error)

        # 드론에 속도를 적용하여 이동 (가상 환경에서만 사용)
        move_drone(drone_speed)

        cv2.imshow('Bottom Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_bottom.release()
    cv2.destroyAllWindows()

    # 사람 인식 스레드 종료
    thread1.join()

# 드론 비행 경로 계획
def plan_flight_path(lane_info):
    # TODO: 인식된 차선 정보를 이용하여 비행 경로 계획
    # 예를 들어, 차선 중앙을 따라 비행하도록 경로를 계획

    # 비행 경로를 계획하여 드론에게 전달
    flight_path = []  # 예시로 빈 리스트 반환, 실제 경로 계획 필요
    return flight_path

# 드론 비행 제어
def control_drone_flight(flight_path):
    # 드론 비행 제어 로직
    for waypoint in flight_path:
        # 드론에게 waypoint로 이동하도록 명령
        # TODO: Roll, Pitch, Yaw를 조절하여 비행 경로를 따라갈 수 있도록 제어

        # 각 waypoint에서 일정 시간 동안 대기
        time.sleep(1)  # 1초 대기 (실제 시나리오에 맞게 조절)

if __name__ == "__main__":
    # 사람 인식과 드론 제어를 멀티스레딩으로 동시에 실행
    thread1 = threading.Thread(target=detect_people)
    thread1.start()

    while True:
        ret, frame = cap_bottom.read()  # 프레임 읽기
        if not ret:
            break

        # TODO: frame을 이용하여 차선 인식 코드 작성
        # 인식된 차선 정보를 lane_info 변수에 저장

        # 비행 경로 계획
        flight_path = plan_flight_path(lane_info)

        # TODO: 드론과 사람 사이의 거리 계산
        distance_error = 0  # 예시로 0으로 초기화, 실제로 거리 계산 필요

        # PID 제어를 이용하여 드론 속도 조절
        drone_speed = control_drone_speed(distance_error)

        # 드론 비행 제어
        control_drone_flight(flight_path)

        cv2.imshow('Bottom Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_bottom.release()
    cv2.destroyAllWindows()

    # 사람 인식 스레드 종료
    thread1.join()