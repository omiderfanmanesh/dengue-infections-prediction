from lightgbm import LGBMClassifier,LGBMRegressor

from model.based import BasedModel
from model.based import TaskMode


class LightGBM(BasedModel):
    def __init__(self, cfg):
        super(LightGBM, self).__init__(cfg=cfg)
        self._task_mode = cfg.BASIC.TASK_MODE

        self._params = {
            'boosting_type': cfg.LIGHTGBM.BOOSTING_TYPE,
            'num_leaves': cfg.LIGHTGBM.NUM_LEAVES,
            'max_depth': cfg.LIGHTGBM.MAX_DEPTH,
            'learning_rate': cfg.LIGHTGBM.LEARNING_RATE,
            'n_estimators': cfg.LIGHTGBM.N_ESTIMATORS,
            'random_state':cfg.LIGHTGBM.RANDOM_STATE
        }

        if self._task_mode == TaskMode.CLASSIFICATION:

            self.model = LGBMClassifier(**self._params)
            self.name = cfg.LIGHTGBM.NAME
            for _k in cfg.LIGHTGBM.HYPER_PARAM_TUNING:
                _param = cfg.LIGHTGBM.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]

        elif self._task_mode == TaskMode.REGRESSION:
            self.model = LGBMRegressor(**self._params)
            self.name = cfg.LIGHTGBM.NAME
            for _k in cfg.LIGHTGBM.HYPER_PARAM_TUNING:
                _param = cfg.LIGHTGBM.HYPER_PARAM_TUNING[_k]
                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]
