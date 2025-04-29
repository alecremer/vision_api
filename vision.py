from ultralytics import YOLO
import cv2
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List
import logging
import numpy as np

@dataclass
class DetectConfig:

    show_video: bool = True
    capture_objects: bool = False
    performance_log: bool = False
    source: str = "self"
    file: str = None
    skip_frames: int = 0
    record: bool = False
    record_file_name: str = None
    ip: str = None
    loop_start_callback: callable = None
    loop_end_callback: callable = None

@dataclass
class DetectModelConfig:

    weights_path: str
    label: str
    confidence: float
    device: str
    segmentation: bool = False


@dataclass
class TrainModelConfig:

    dataset_path: str
    epochs: int
    label: str
    device: str
    model: str
    results_folder_name: str

class Train:

    @classmethod
    def train_one_model(self, train_cfg: TrainModelConfig):
        
        # parse
        epochs = train_cfg.epochs
        path = train_cfg.dataset_path
        device = train_cfg.device
        model = train_cfg.model
        results_folder_name = train_cfg.results_folder_name

        model = YOLO("models/" + model)

        results = model.train(data=(path + "/data.yaml"), epochs=epochs, device=device, project=path + "/runs", name=results_folder_name)
        model.val()

    @classmethod
    def train(self, train_cfg_list: List[TrainModelConfig]):

        if train_cfg_list and len(train_cfg_list) > 0:
            
            for cfg in train_cfg_list:

                self.train_one_model(cfg)

        else:
            raise("train configuration could not be empty")

class Vision:

    _train = Train()

    def __init__(self):
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

        pass

    def train(self, train_cfg_list: List[TrainModelConfig]): 
        self._train.train(train_cfg_list)
    

    def test(self, weight_path, test_path, show = True):
        
        model_trained = YOLO(weight_path)
        result = model_trained.predict(test_path, show=show)[0]

    def _select_cam_source_(self, source, ip=None, file=None):

        if source == "self":
            
            cam = cv2.VideoCapture(0)
        
        elif source == "rtsp":
            
            print("run rtsp ip: " + ip)
            cam = cv2.VideoCapture("rtsp://" + ip)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cam.set(cv2.CAP_PROP_FPS, 30)

        elif source == "file":
            cam = cv2.VideoCapture(file)

        return cam
    
    def _set_trained_models(self, weight_paths):

        models_trained = []

        for p in weight_paths:
            model_trained = YOLO(p)
            model_trained.verbose = False
            models_trained.append(model_trained)
        
        if not models_trained:
            raise Exception("weights paths are empty")

        return models_trained


    def _live_detection_loop(self, cam, confidence, labels, segmentation, models_trained, config: DetectConfig):

        result = None
        
        frame_count = 0

        process_frame = False
        frame_stream = []
        
        while True:
            

            if config.skip_frames != 0:
                
                if frame_count % config.skip_frames == 0:

                    process_frame = True
                    
                    frame_count = 0
                else:
                    process_frame = False

                frame_count += 1
            else:
                
                process_frame = True
              
            index = 0

                    
            check, frame = cam.read()
            if process_frame:


                if config.loop_start_callback:
                    config.loop_start_callback()
                
                for m in models_trained:
                    # result = m(frame, stream=True, conf=0.65)
                    result = m(frame, stream=True, conf=confidence[index])
                    
                    if config.capture_objects:
                        objects = []
                        for r in result:
                            boxes = r.boxes
                            # for box in boxes:
                            #     objects.append(box.xyxy[0]) 
                    if config.show_video:

                        if segmentation[index]:
                            frame = self.create_masks_in_frames(result, frame, labels[index])
                        else:
                            self.create_bounding_box(result, frame, labels[index])
                        
                            

                    index = index + 1

                if config.show_video:
                    frame_stream.append(frame)
                    cv2.imshow('video', frame)

                if config.loop_end_callback:
                    config.loop_end_callback()



            key = cv2.waitKey(1)
            if key == 27: # esc 
                
                if(config.record):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de vídeo
                    output_file = config.record_file_name + ".mkv"  # Nome do arquivo de saída
                    fps = 10.0  # Quadros por segundo
                    frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)  # Cria o objeto de gravação de vídeo
                    for f in frame_stream:
                        out.write(f)

                    out.release()
                break

    def live_detection(self, detect_model_config: List[DetectModelConfig], config: DetectConfig):
        


        weight_paths = []
        labels = []
        confidence = []
        segmentation = []

        print("detection for classe: ")
        for model_cfg in detect_model_config:

            weight_paths.append(model_cfg.weights_path)
            labels.append(model_cfg.label)
            confidence.append(model_cfg.confidence)
            segmentation.append(model_cfg.segmentation)

            print(model_cfg.label)

        models_trained = self._set_trained_models(weight_paths)
        cam = self._select_cam_source_(config.source, config.ip, config.file)

        self._live_detection_loop(cam, confidence, labels, segmentation, models_trained, config)
        


        cam.release()

        cv2.destroyAllWindows()

    def create_bounding_box(self, detection_result, frame, label):
        for r in detection_result:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, label + " " + str(confidence), org, font, fontScale, color, thickness)

    def create_masks_in_frames(self, result, frame, label):
        
        for r in result:
            if r.masks is not None and len(r.masks.data) > 0:
                
                mask = r.masks.data[0].cpu().numpy()
                if mask is not None and mask.size > 0:
                    # Redimensionando a máscara para o tamanho do frame original
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # Convertendo a máscara para binário (0 ou 255)
                    mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    mask_colored = np.zeros_like(frame, dtype=np.uint8)
                    mask_colored[:, :, 1] = mask_resized  # Aplicando verde na máscara
                    alpha = 0.5  # Nível de transparência
                    frame = cv2.addWeighted(frame, 1, mask_colored, alpha, 0)

                    # Encontrar o contorno da máscara
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # Filtro para contornos pequenos
                            # Obtendo as coordenadas do retângulo delimitador
                            x, y, w, h = cv2.boundingRect(contour)
                            x1, y1 = x, y
                            x2, y2 = x + w, y + h
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2
                            confidence_mask = r.boxes.conf[0].item() if r.boxes is not None and len(r.boxes) > 0 else 0.0
                            confidence_mask =  math.ceil((confidence_mask*100))/100
                            cv2.putText(frame, label + " " + str(confidence_mask), org, font, fontScale, color, thickness)

        return frame