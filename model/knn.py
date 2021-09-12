from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from model.based import BasedModel
from model.based import TaskMode

class KNN(BasedModel):
    def __init__(self,cfg):
        super(KNN, self).__init__(cfg=cfg)
        self._task_mode = cfg.BASIC.TASK_MODE
        if self._task_mode == TaskMode.CLASSIFICATION:
            self._params = {
                'n_neighbors': cfg.KNNC.N_NEIGHBORS,
                'weights': cfg.KNNC.WEIGHTS,
                'algorithm': cfg.KNNC.ALGORITHM,
                'leaf_size': cfg.KNNC.LEAF_SIZE,
                'p': cfg.KNNC.P,
                'metric': cfg.KNNC.METRIC,
                'metric_params': cfg.KNNC.METRIC_PARAMS,
                'n_jobs': cfg.KNNC.N_JOBS,
            }
            self.model = KNeighborsClassifier(**self._params)
            self.name = cfg.KNNC.NAME
            for _k in cfg.KNNC.HYPER_PARAM_TUNING:
                _param = cfg.KNNC.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]

        elif self._task_mode == TaskMode.REGRESSION:
            self._params = {
                'n_neighbors': cfg.KNNR.N_NEIGHBORS,
                'weights': cfg.KNNR.WEIGHTS,
                'algorithm': cfg.KNNR.ALGORITHM,
                'leaf_size': cfg.KNNR.LEAF_SIZE,
                'p': cfg.KNNR.P,
                'metric': cfg.KNNR.METRIC,
                'metric_params': cfg.KNNR.METRIC_PARAMS,
                'n_jobs': cfg.KNNR.N_JOBS,
            }
            self.model = KNeighborsRegressor(**self._params)
            self.name = cfg.KNNR.NAME
            for _k in cfg.KNNR.HYPER_PARAM_TUNING:
                _param = cfg.KNNR.HYPER_PARAM_TUNING[_k]
                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]