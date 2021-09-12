from xgboost import XGBClassifier, XGBRegressor

from model.based import BasedModel
from model.based import TaskMode


class XGBOOST(BasedModel):
    def __init__(self, cfg):
        super(XGBOOST, self).__init__(cfg=cfg)
        self._task_mode = cfg.BASIC.TASK_MODE

        self._params = {
            'n_estimators': cfg.XGBOOST.N_ESTIMATORS,
            'max_depth': cfg.XGBOOST.MAX_DEPTH,
            'learning_rate': cfg.XGBOOST.LEARNING_RATE,
            'verbosity': cfg.XGBOOST.VERBOSITY,
            'objective': cfg.XGBOOST.OBJECTIVE,
            'booster': cfg.XGBOOST.BOOSTER,
            'tree_method': cfg.XGBOOST.TREE_METHOD,
            'n_jobs': cfg.XGBOOST.N_JOBS,

            'gamma': cfg.XGBOOST.GAMMA,
            'min_child_weight': cfg.XGBOOST.MIN_CHILD_WEIGHT,
            'max_delta_step': cfg.XGBOOST.MAX_DELTA_STEP,
            'subsample': cfg.XGBOOST.SUBSAMPLE,
            'colsample_bytree': cfg.XGBOOST.COLSAMPLE_BYTREE,
            'colsample_bylevel': cfg.XGBOOST.COLSAMPLE_BYLEVEL,
            'colsample_bynode': cfg.XGBOOST.COLSAMPLE_BYNODE,
            'reg_alpha': cfg.XGBOOST.REG_ALPHA,

            'reg_lambda': cfg.XGBOOST.REG_LAMBDA,
            'scale_pos_weight': cfg.XGBOOST.SCALE_POS_WEIGHT,
            'base_score': cfg.XGBOOST.BASE_SCORE,
            'random_state': cfg.XGBOOST.RANDOM_STATE,
            'missing': cfg.XGBOOST.MISSING,
            'num_parallel_tree': cfg.XGBOOST.NUM_PARALLEL_TREE,
            'monotone_constraints': cfg.XGBOOST.MONOTONE_CONSTRAINTS,
            'interaction_constraints': cfg.XGBOOST.INTERACTION_CONSTRAINTS,

            'importance_type': cfg.XGBOOST.IMPORTANCE_TYPE,
            'gpu_id': cfg.XGBOOST.GPU_ID,
            'validate_parameters': cfg.XGBOOST.VALIDATE_PARAMETERS,

        }

        if self._task_mode == TaskMode.CLASSIFICATION:

            self.model = XGBClassifier(**self._params)
            self.name = cfg.XGBOOST.NAME
            for _k in cfg.XGBOOST.HYPER_PARAM_TUNING:
                _param = cfg.XGBOOST.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]

        elif self._task_mode == TaskMode.REGRESSION:
            self.model = XGBRegressor(**self._params)
            self.name = cfg.XGBOOST.NAME
            for _k in cfg.XGBOOST.HYPER_PARAM_TUNING:
                _param = cfg.XGBOOST.HYPER_PARAM_TUNING[_k]
                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]
