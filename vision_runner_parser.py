import yaml
from typing import List, Tuple
from vision import DetectModelConfig, TrainModelConfig, AnnotateModelConfig


class VisionRunnerParser:

    def parse_from_file(self, file_name_or_path="config.yaml"):

        print(file_name_or_path)
        configs = self._get_configs_from_file(file_name_or_path)
        (train_config_dict, detect_config_dict, annotate_dict) = self._parse_configs(configs)
        train_cfg, detect_cfg = self._config_to_list_of_instances(detect_config_dict, train_config_dict)
        annotate_cfg = self._config_to_list_of_annotate(annotate_dict)
        
        return train_cfg, detect_cfg, annotate_cfg


    def _get_configs_from_file(self, file_name_or_path="config.yaml") -> List[any]:

        with open(file_name_or_path) as file_name:
            try:
                all_configs = yaml.safe_load(file_name)
            except:
                raise("cannot open config file")
            
        return all_configs
    
    def _parse_configs(self, configs: List[any]) -> Tuple[dict, dict, dict]:
        '''Return tuple of train_config_dict and detect_config_dict'''
        train_config_dict = []
        detect_config_dict = []
        annotate_config_dict = []
        
        for cfg in configs:

            if configs[cfg]["train"]:
                train_config_dict.append(configs[cfg])

            
            if configs[cfg]["detect"]:
                detect_config_dict.append(configs[cfg])

            if configs[cfg]["annotate"]:
                annotate_config_dict.append(configs[cfg])

        return train_config_dict, detect_config_dict, annotate_config_dict
    
    def _config_to_list_of_instances(self, detect_config_dict: dict, train_config_dict: dict) -> Tuple[List[TrainModelConfig], List[DetectModelConfig]]:
        train_cfg = []
        detect_cfg = []

        for detect in detect_config_dict:

            detect_cfg.append(DetectModelConfig(detect["weights"], detect["label"], detect["confidence"], detect["device"], detect.get("segmentation", False)))

        for train in train_config_dict:

            train_cfg.append(TrainModelConfig(train["dataset"], train["epochs"], train["label"], train["device"],
                                        train["model"], train["result_folder_name"]))
            
        return train_cfg, detect_cfg

    def _config_to_list_of_annotate(self, annotate_config_dict: dict) -> List[AnnotateModelConfig]:
        annotate_cfg = []

        for annotate in annotate_config_dict:
            annotate_cfg.append(AnnotateModelConfig(annotate["weights"], annotate["labels_to_annotate"], annotate["annotate_confidence"]))

        return annotate_cfg