import cv2
import numpy as np

# Configurar os caminhos corretos para o YOLO
# Entre "/Users/" é "/Desktop/" addicione o nome user
config_path = 'C:/Users//Desktop/yolo/yolov3.cfg'
weights_path = 'C:/Users//Desktop/yolo/yolov3.weights'
names_path = 'C:/Users//Desktop/yolo/coco.names'

# Carregar o modelo YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar os nomes das classes
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)  # 0 para a webcam padrão

def draw_detection(frame, x, y, w, h, label, color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def recognize_face(frame, x, y, w, h):
    face_name = "                 Humano"  # Espaço reservado para reconhecimento simulado
    return face_name

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preparar o quadro para o YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Processar a saída do YOLO
    class_ids, confidences, boxes = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supressão não máxima para eliminar caixas sobrepostas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            object_name = classes[class_ids[i]]
            
            # Substituir "Cell Phone" por "Smartphone"
            if object_name == "cell phone":
                object_name = "Smartphone"
            
            label = f"{object_name}: {confidences[i]:.2f}"
            draw_detection(frame, x, y, w, h, label)

            # Se a classe detectada for "pessoa", aplicar reconhecimento facial
            if classes[class_ids[i]] == "person":
                face_label = recognize_face(frame, x, y, w, h)
                draw_detection(frame, x, y, w, h, face_label, color=(255, 0, 0))

    # Exibir o quadro com as detecções
    cv2.imshow("YOLO Object Detection", frame)

    # Pressionar 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o capturador de vídeo e fechar janelas
cap.release()
cv2.destroyAllWindows()
