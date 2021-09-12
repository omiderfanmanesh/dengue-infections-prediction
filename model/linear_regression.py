from sklearn.linear_model import LinearRegression as LR

from model.based import BasedModel


class LinearRegression(BasedModel):
    def __init__(self, cfg):
        super(LinearRegression, self).__init__(cfg=cfg)
        self._params = {
            'fit_intercept': cfg.LR.FIT_INTERCEPT,
            'normalize': cfg.LR.NORMALIZE,
            'copy_X': cfg.LR.COPY_X,
            'n_jobs': cfg.LR.N_JOBS,
        }
        self.model = LR()
        self.name = cfg.LR.NAME
        for _k in cfg.LR.HYPER_PARAM_TUNING:
            _param = cfg.LR.HYPER_PARAM_TUNING[_k]
            if _param is not None:
                _param = [*_param]
                self.fine_tune_params[_k.lower()] = [*_param]
