from vision import Vision, DetectModelConfig, TrainModelConfig, DetectConfig
from vision_runner_parser import VisionRunnerParser

class VisionRunner:

    def live(self, detect_cfg: DetectConfig, file_config_or_path: str = "config.yaml"):

        model = Vision()
        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)
        model.live_detection(detect_model_cfg, detect_cfg)    

    def train(self, file_config_or_path: str = "config.yaml"):

        model = Vision()
        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)
        model.train(train_model_cfg)

    def annotate(self, img_path: str, file_config_or_path: str = "config.yaml"):
        model = Vision()
        train_model_cfg, detect_model_cfg, annotate_cfg = VisionRunnerParser().parse_from_file(file_config_or_path)
        model.annotate(img_path, annotate_cfg)

    # def get_models(file_config_or_path: str = "config.yaml")
