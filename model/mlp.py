from sklearn.neural_network import MLPClassifier,MLPRegressor 
from model.based import BasedModel
from model.based import TaskMode


class MLP(BasedModel):
    def __init__(self,cfg):
        super(MLP, self).__init__(cfg=cfg)
        self._task_mode = cfg.BASIC.TASK_MODE

        self._params = {
            'hidden_layer_sizes': cfg.MLP.HIDDEN_LAYER_SIZES,
            'activation': cfg.MLP.ACTIVATION,
            'solver': cfg.MLP.SOLVER,
            'alpha': cfg.MLP.ALPHA,
            'batch_size': cfg.MLP.BATCH_SIZE,
            'learning_rate': cfg.MLP.LEARNING_RATE,
            'learning_rate_init': cfg.MLP.LEARNING_RATE_INIT,
            'power_t': cfg.MLP.POWER_T,

            'max_iter': cfg.MLP.MAX_ITER,
            'shuffle': cfg.MLP.SHUFFLE,
            'random_state': cfg.MLP.RANDOM_STATE,
            'tol': cfg.MLP.TOL,
            'verbose': cfg.MLP.VERBOSE,
            'warm_start': cfg.MLP.WARM_START,
            'momentum': cfg.MLP.MOMENTUM,
            'nesterovs_momentum': cfg.MLP.NESTEROVS_MOMENTUM,

            'early_stopping': cfg.MLP.EARLY_STOPPING,
            'validation_fraction': cfg.MLP.VALIDATION_FRACTION,
            'beta_1': cfg.MLP.BETA_1,
            'beta_2': cfg.MLP.BETA_2,
            'epsilon': cfg.MLP.EPSILON,
            'n_iter_no_change': cfg.MLP.N_ITER_NO_CHANGE,
            'max_fun': cfg.MLP.MAX_FUN,

        }
        
        
        if self._task_mode == TaskMode.CLASSIFICATION:
            self.model = MLPClassifier(**self._params)
            self.name = cfg.MLP.NAME
            for _k in cfg.MLP.HYPER_PARAM_TUNING:
                _param = cfg.MLP.HYPER_PARAM_TUNING[_k]

                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]

        elif self._task_mode == TaskMode.REGRESSION:

            self.model = MLPRegressor(**self._params)
            self.name = cfg.MLP.NAME
            for _k in cfg.MLP.HYPER_PARAM_TUNING:
                _param = cfg.MLP.HYPER_PARAM_TUNING[_k]
                if _param is not None:
                    _param = [*_param]
                    self.fine_tune_params[_k.lower()] = [*_param]