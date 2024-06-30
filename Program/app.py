import cv2
import numpy as np
import torch
import time

# YOLOv5 modelini yükleme kısmı
#eğer küçük nesneleride algılamak istiyorsanız yolov5s kullanabilirsiniz ben tercih etmedim çünkü alakasız sonuçlar çıkıyor.
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

#Burada güven eşliğini arttırdım ve birazda olsa yanlış eşleşmeleri azalttım.
def detect_objects(frame, confidence_threshold=0.5):
    results = model(frame)
    labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    detections = []
    for i in range(len(labels)):
        row = cords[i]
        if row[4] >= confidence_threshold:
            detections.append((int(labels[i]), row))
    return detections



cap = cv2.VideoCapture(0)

detections = []
frame_count = 0
wait_frames = 5  # Her nesne algılamasını belirli sayıda kare boyunca bekletiyorum

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % wait_frames == 0:
        # Nesneleri algılama bölümü
        new_detections = detect_objects(frame, confidence_threshold=0.5)
        
        # Geçerli algılamalarla eşleştir
        detections = new_detections
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    
    for label, row in detections:
        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
        bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, model.names[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    
    cv2.imshow('Video', frame)
    #Programı sonlandırmak için buradan tuş atayabilirsiniz ben adımın baş harfini yaptım
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

cap.release()
cv2.destroyAllWindows()
