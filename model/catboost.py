from catboost import CatBoostClassifier,CatBoostRegressor

from model.based import BasedModel
from model.based import TaskMode


class CatBoost(BasedModel):
    def __init__(self, cfg):
        super(CatBoost, self).__init__(cfg=cfg)
        self._task_mode = cfg.BASIC.TASK_MODE

        self._params = {
            # 'iterations': cfg.CATBOOST.ITERATIONS,
            'learning_rate': cfg.CATBOOST.LEARNING_RATE,
            'depth': cfg.CATBOOST.DEPTH,
            'l2_leaf_reg': cfg.CATBOOST.L2_LEAF_REG,
            'loss_function': cfg.CATBOOST.LOSS_FUNCTION,
            'random_seed':cfg.CATBOOST.RANDOM_SEED,

            'use_best_model': cfg.CATBOOST.USE_BEST_MODEL,
            'verbose': cfg.CATBOOST.VERBOSE,
            'eval_metric': cfg.CATBOOST.EVAL_METRIC,
            'boosting_type': cfg.CATBOOST.BOOSTING_TYPE,
            'task_type': cfg.CATBOOST.TASK_TYPE,
            'n_estimators': cfg.CATBOOST.N_ESTIMATORS,
            'cat_features': cfg.CATBOOST.CAT_FEATURES,
        }

        if self._task_mode == TaskMode.CLASSIFICATION:

            self.model = CatBoostClassifier(**self._params)
            self.name = cfg.CATBOOST.NAME
            for _k in cfg.CATBOOST.HYPER_PARAM_TUNING:
                _param = cfg.CATBOOST.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]

        elif self._task_mode == TaskMode.REGRESSION:
            self.model = CatBoostRegressor(**self._params)
            self.name = cfg.CATBOOST.NAME
            for _k in cfg.CATBOOST.HYPER_PARAM_TUNING:
                _param = cfg.CATBOOST.HYPER_PARAM_TUNING[_k]
                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]