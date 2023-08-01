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

# VehicleMode를 'GUIDED'로 설정
vehicle.mode = VehicleMode('GUIDED')

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
# 비디오 캡처 객체 생성
def create_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

# 머리 위치를 픽셀 단위로 검출하는 함수
def detect_head_position(frame):
    # 색공간 변환 (BGR -> HSV)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 헤어밴드의 색상 범위 (HSV 기준)
    lower_red = (0, 100, 100)
    upper_red = (10, 255, 255)

    # 빨간색 헤어밴드를 마스크로 추출
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # 모폴로지 연산을 통해 마스크 세부 처리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 윤곽 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 머리 위치 추정
    head_center_x = 0
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            head_center_x = int(moments['m10'] / moments['m00'])

    return head_center_x

# 화면 중앙으로부터 머리의 위치를 계산하여 distance_error 값을 생성하는 함수
def get_distance_error(head_center_x, image_width):
    # 화면 중앙의 x 좌표
    screen_center_x = image_width // 2

    # 머리의 위치와 화면 중앙과의 거리 계산
    distance_from_center = head_center_x - screen_center_x

    # distance_error 값을 생성 (머리가 오른쪽에 있으면 음수, 왼쪽에 있으면 양수)
    distance_error = -distance_from_center

    return distance_error

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

# 테스트 비디오 경로
video_path = 'test_video.mp4'

# 비디오 캡처 객체 생성
cap = create_video_capture(video_path)

# 비디오 프레임의 너비
image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():
    # 비디오 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 머리 위치 검출
    head_center_x = detect_head_position(frame)

    # 머리 위치를 기반으로 distance_error 값을 계산합니다.
    distance_error = get_distance_error(head_center_x, image_width)

    # control_drone_speed 함수를 호출하여 드론의 속도를 제어합니다.
    drone_speed = control_drone_speed(distance_error)
    print("드론 속도 조절값:", drone_speed)

    # 화면에 결과 영상 표시 (테스트용, 실제로는 필요 없을 수 있음)
    cv2.imshow('Head Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 'ESC' 키를 누르면 종료
        break

# 비디오 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()


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
    # 차선 중앙을 따라 비행하는 비행 경로를 계획합니다.
    flight_path = []  # 비행 경로를 저장할 빈 리스트를 생성합니다.

    if lane_info is not None and len(lane_info) > 0:
        # 인식된 차선 정보가 유효한 경우에만 처리합니다.

        # 차선 중앙을 계산하기 위해 좌우 차선의 위치를 확인합니다.
        left_lane = lane_info['left']  # 좌측 차선 정보
        right_lane = lane_info['right']  # 우측 차선 정보

        if left_lane is not None and right_lane is not None:
            # 좌우 차선 모두 인식된 경우에만 처리합니다.

            # 좌우 차선 중앙을 계산합니다.
            center_lane = (left_lane + right_lane) / 2

            # 드론이 차선 중앙을 따라가도록 비행 경로를 생성합니다.
            for y in range(len(center_lane)):
                x = center_lane[y]  # 해당 y 좌표에서의 차선 중앙 값
                flight_path.append((x, y))  # (x, y) 좌표를 비행 경로에 추가합니다.

    return flight_path

# 드론 비행 제어
def control_drone(flight_path):
    # 드론 비행 제어 로직
    for waypoint in flight_path:
        x, y = waypoint  # waypoint의 x, y 좌표 추출

        # 드론에게 waypoint로 이동하도록 명령
        move_to_waypoint(x, y)

        # Roll, Pitch, Yaw 값을 계산합니다.
        roll, pitch, yaw = calculate_roll_pitch_yaw(x, y)

        # 드론을 Roll, Pitch, Yaw 값으로 제어합니다.
        set_roll_pitch_yaw(roll, pitch, yaw)

        # 각 waypoint에서 일정 시간 동안 대기
        time.sleep(1)  # 1초 대기 (실제 시나리오에 맞게 조절)

def move_to_waypoint(x, y):
    # 드론에게 waypoint로 이동하도록 명령하는 함수
    # 예시로써, 간단히 출력만 수행합니다.
    print(f"Move to waypoint: x={x}, y={y}")

def calculate_roll_pitch_yaw(x, y):
    # 드론의 Roll, Pitch, Yaw 값을 계산하는 함수
    # 실제로는 비행 경로와 드론의 현재 상태에 따라 Roll, Pitch, Yaw 값을 계산해야 합니다.
    # 예시로써, 일정한 Roll, Pitch 값을 반환합니다.
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    return roll, pitch, yaw

def set_roll_pitch_yaw(roll, pitch, yaw):
    # 드론의 Roll, Pitch, Yaw 값을 설정하는 함수
    # 실제 드론과 연동하여 Roll, Pitch, Yaw 값을 설정해야 합니다.
    # 예시로써, 간단히 출력만 수행합니다.
    print(f"Set Roll={roll}, Pitch={pitch}, Yaw={yaw}")

# 위의 control_drone 함수를 호출하여 드론을 비행 경로를 따라가도록 제어합니다.
# 예를 들어,
flight_path = [(10, 20), (15, 25), (20, 30)]  # 예시 비행 경로
control_drone(flight_path)

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
