import cv2
import numpy as np
from dronekit import connect

# Drone 연결 설정 (필요시 주소를 변경)
# connection_string = 'udp:127.0.0.1:14551'
# vehicle = connect(connection_string, wait_ready=True)

# 포즈 추정을 위한 OpenPose 모델 불러오기
net = cv2.dnn.readNetFromCaffe('pose_deploy.prototxt', 'pose_iter_440000.caffemodel')

# 스켈레톤을 그릴 포인트와 연결
body_parts = {
    0: 'Head',
    1: 'Neck',
    2: 'Shoulder L',
    3: 'Elbow L',
    4: 'Wrist L',
    5: 'Shoulder R',
    6: 'Elbow R',
    7: 'Wrist R',
    8: 'Hip L',
    9: 'Knee L',
    10: 'Ankle L',
    11: 'Hip R',
    12: 'Knee R',
    13: 'Ankle R',
}

# 연결 정보
connections = [
    (1, 2), (2, 3), (3, 4),  # 왼쪽 팔
    (1, 5), (5, 6), (6, 7),  # 오른쪽 팔
    (1, 8), (8, 9), (9, 10),  # 왼쪽 다리
    (1, 11), (11, 12), (12, 13),  # 오른쪽 다리
]

# 비디오 파일 또는 카메라 소스 설정
cap = cv2.VideoCapture('path_to_your_video.mp4')  # 비디오 파일 경로

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 이미지 전처리
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.01, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 포즈 추정 결과 얻기
    output = net.forward()
    
    # 포즈 정보 저장
    points = []
    for i in range(len(body_parts)):
        # 예측된 점의 신뢰도
        prob_map = output[0, i]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        
        if prob > 0.1:  # 신뢰도가 임계값 이상인 경우
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # 스켈레톤 그리기
    for connection in connections:
        partA = connection[0]
        partB = connection[1]
        
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)
            cv2.circle(frame, points[partA], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[partB], 5, (0, 0, 255), -1)

    # 결과 영상 표시
    cv2.imshow('Skeleton', frame)
    
    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
# vehicle.close()  # 드론 연결 해제 (사용 시)
