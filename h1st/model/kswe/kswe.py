from typing import Dict, List
from datetime import datetime
import ulid

import pandas as pd

from h1st.model.predictive_model import PredictiveModel
from h1st.model.model import Model

class KSWE(PredictiveModel):
    def __init__(self):
        super().__init__()
        self.stats = {}
        self.segmentor = None
        self.sub_models = None
        self.ensemble = None

    def predict(self, input_data: Dict):
        # Segment data
        segmented_data = self.segmentor.process(input_data)['segment_data']

        # Generate sub_models' prediction
        sub_model_predictions = {
            name: sub_model.predict({'X': segmented_data[name]})['predictions']
            for name, sub_model in self.sub_models.items()
        }

        # Generage ensemble's prediction
        final_predictions = self.ensemble.predict(sub_model_predictions)

        return final_predictions

    def persist(self, version: str=None) -> None:
        if version is None:
            now = datetime.utcnow()
            now_str = now.strftime('%Y%m%d-%H%M')
            version = f'{str(ulid.new())[:4]}-{now_str}'

        self.stats['version'] = version
        self.stats['ensemble_version'] = self.ensemble.persist(version)
        self.stats['ensemble_class'] = self.ensemble.__class__
        self.stats['segmentor_version'] = self.segmentor.persist(version)
        self.stats['segmentor_class'] = self.segmentor.__class__

        sub_model_info = {}
        for name, sub_model in self.sub_models.items():
            v = sub_model.persist(version+f'_{name}')
            sub_model_info[name] = {'version': v, 'model_class': sub_model.__class__}
        self.stats['sub_model_info'] = sub_model_info
        super().persist(version)
        return version

    def load_params(self, version: str=None) -> None:
        super().load_params(version)
        ensemble_class = self.stats['ensemble_class']
        ensemble_version = self.stats['ensemble_version']
        self.ensemble = ensemble_class().load_params(ensemble_version)

        segmentor_class = self.stats['segmentor_class']
        segmentor_version = self.stats['segmentor_version']
        self.segmentor = segmentor_class().load_params(segmentor_version)

        sub_models = {}
        for name, info in self.stats['sub_model_info'].items():
            sub_model = info['model_class']().load_params(info['version'])
            sub_models[name] = sub_model
        self.sub_models = sub_models
        return self

    @classmethod
    def construct_model(cls, segmentor: Model, sub_models: dict, ensemble: Model):
        model = cls()
        model.segmentor = segmentor
        model.sub_models = sub_models
        model.ensemble = ensemble
        return model


    # def persist(self, version: str) -> None:
    #     self.ensemble.persist(version+'_ensemble')
    #     sub_model_names = []
    #     for name, sub_model in self.sub_models.items():
    #         sub_model.persist(version+f'_{name}')
    #         sub_model_names.append(name)
    #     self.stats['sub_model_names'] = sub_model_names
    #     super().persist(version)

    # def load_params(self, version: str) -> None:
    #     super().load_params(version)
    #     sub_model_names = self.stats['sub_model_names']
    #     self.ensemble.load_params(version+'_ensemble')
    #     sub_models = {}
    #     for name in sub_model_names:
    #         sub_model = self.sub_model_class().load_params(version+f'_{name}')
    #         sub_models[name] = sub_model
    #     self.sub_models = sub_models


