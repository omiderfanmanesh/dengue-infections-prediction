#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from yacs.config import CfgNode as CN

from data.based import EncoderTypes, ScaleTypes, TransformersType
from model.based import MetricTypes, TaskMode
from model.based import Model
from utils import RuntimeMode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# BASIC CONFIG
# -----------------------------------------------------------------------------
_C.BASIC = CN()
_C.BASIC.SEED = 2021
_C.BASIC.PCA = True  # pca = True will apply principal component analysis to data
_C.BASIC.TRANSFORMATION = True
_C.BASIC.RAND_STATE = 2021
_C.BASIC.OUTPUT = '../output/'
_C.BASIC.SAVE_MODEL = '../output/model.joblib'
_C.BASIC.MODEL = Model.KNN  # select training model e.g. SVM, RandomForest, ...
_C.BASIC.RUNTIME_MODE = RuntimeMode.TRAIN  # runtime modes {Train, cross validation, hyperparameter tuning}
_C.BASIC.TASK_MODE = TaskMode.REGRESSION  # task mode = {classification, regression}
# data resampling, {None,SMOTE,RANDOM_UNDER_SAMPLING} and {None} means don't use
# resampling, order is important e.g. (Sampling.SMOTE, Sampling.RANDOM_UNDER_SAMPLING),
_C.BASIC.SAMPLING_STRATEGY = None
# _C.BASIC.SAMPLING_STRATEGY = (Sampling.RANDOM_UNDER_SAMPLING)
# _C.BASIC.SAMPLING_STRATEGY = (Sampling.RANDOM_OVER_SAMPLING, Sampling.RANDOM_UNDER_SAMPLING)
# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 2  # number of target classes for classification task
_C.MODEL.K_FOLD = 5  # value of K for KFold cross-validation
_C.MODEL.SHUFFLE = True  # shuffle the data for cross-validation task

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET_ADDRESS = '../data/dataset/development.tsv'  # the address of dataset file
_C.DATASET.EVALUATION_DATASET_ADDRESS = '../data/dataset/evaluation.tsv'  # the address of dataset file
_C.DATASET.DATASET_BRIEF_DESCRIPTION = None  # if you have a brief description for your dataset
_C.DATASET.TARGET = 'total_cases'  # target column of your dataset
_C.DATASET.HAS_CATEGORICAL_TARGETS = False  # True = if you have a categorical targets otherwise False

"""
('year', 'weekofyear', 'week_start_date', 'PERSIANN_precip_mm',
       'NCEP_air_temp_k', 'NCEP_avg_temp_k', 'NCEP_dew_point_temp_k',
       'NCEP_max_air_temp_k', 'NCEP_min_air_temp_k', 'NCEP_precip_kg_per_m2',
       'NCEP_humidity_percent', 'NCEP_precip_mm', 'NCEP_humidity_g_per_kg',
       'NCEP_diur_temp_rng_k', 'avg_temp_c', 'diur_temp_rng_c', 'max_temp_c',
       'min_temp_c', 'precip_mm', 'total_cases', 'city')
"""

# columns that you need to drop from dataframe
_C.DATASET.DROP_COLS = (

    # 'year',
    'weekofyear',
    'week_start_date',
    'PERSIANN_precip_mm',
    'NCEP_air_temp_k',
    'NCEP_avg_temp_k',
    'NCEP_dew_point_temp_k',
    'NCEP_max_air_temp_k',
    'NCEP_min_air_temp_k',
    'NCEP_precip_kg_per_m2',
    'NCEP_humidity_percent',
    'NCEP_precip_mm',
    'NCEP_humidity_g_per_kg',
    'NCEP_diur_temp_rng_k',
    'avg_temp_c',
    'diur_temp_rng_c',
    'max_temp_c',
    'min_temp_c',
    'precip_mm',
    # 'total_cases',
    # 'city',
    # 'season',
    'month',
    # "six",
    # 'week_split',
    # 'weekofyear_sin',
    # 'weekofyear_cos',
    'month_sin',
    'month_cos',
    'quarter',
    'PERSIANN_precip_mm_no_nans',
    'NCEP_avg_temp_c',
    'NCEP_avg_temp_c_no_nans',
    'NCEP_diur_temp_rng_c',
    'NCEP_diur_temp_rng_c_no_nans',
    'NCEP_max_air_temp_c',
    'NCEP_max_air_temp_c_no_nans',
    'NCEP_min_air_temp_c',
    'NCEP_min_air_temp_c_no_nans',
    'NCEP_air_temp_c',
    'NCEP_air_temp_c_no_nans',
    'NCEP_dew_point_temp_c',
    'NCEP_dew_point_temp_c_no_nans',
    'avg_temp_c_no_nans',
    'diur_temp_rng_c_no_nans',
    'max_temp_c_no_nans',
    'min_temp_c_no_nans',
    'precip_mm_no_nans'
)
# ----------------------------------------------------------------------------
# metric
# ----------------------------------------------------------------------------
_C.EVALUATION = CN()

_C.EVALUATION.METRIC = MetricTypes.MEAN_ABSOLUTE_ERROR  # select your metric for your model
_C.EVALUATION.CONFUSION_MATRIX = False  # set True if you need to plot the confusion matrix

"""
Supported metrics:

'accuracy', 'balanced_accuracy',  'top_k_accuracy',
 'average_precision',  'neg_brier_score', 'f1',
 'f1_micro', 'f1_macro',  'f1_weighted',
 'f1_samples',  'neg_log_loss', 'precision',
  'recall',  'jaccard', 'roc_auc',
 'roc_auc_ovr', 'roc_auc_ovo',  'roc_auc_ovr_weighted',
 'roc_auc_ovo_weighted'
 
"""
# -----------------------------------------------------------------------------
# CATEGORICAL FEATURES ENCODER CONFIG / _C.ENCODER.{COLUMN NAME} = TYPE OF ENCODER
# -----------------------------------------------------------------------------
# if you have categorical column, write its name in CAPITAL letter
_C.ENCODER = CN()
_C.ENCODER.CITY = EncoderTypes.BINARY
# -----------------------------------------------------------------------------
# SCALER
# -----------------------------------------------------------------------------
_C.SCALER = CN()
_C.SCALER = ScaleTypes.STANDARD  # select the type of scaler (STANDARD SCALER, MINMAX SCALER, ...) that you want to apply to your data

# -----------------------------------------------------------------------------
# TRANSFORMATION
# -----------------------------------------------------------------------------
_C.TRANSFORMATION = CN()
_C.TRANSFORMATION.PERSIANN_PRECIP_MM_NO_NANS = TransformersType.LOG
_C.TRANSFORMATION.NCEP_AVG_TEMP_C_NO_NANS = TransformersType.LOG
_C.TRANSFORMATION.NCEP_DIUR_TEMP_RNG_C_NO_NANS = TransformersType.LOG
_C.TRANSFORMATION.NCEP_MAX_AIR_TEMP_C_NO_NANS = TransformersType.LOG
_C.TRANSFORMATION.NCEP_MIN_AIR_TEMP_C_NO_NANS = TransformersType.LOG
_C.TRANSFORMATION.NCEP_AIR_TEMP_C_NO_NANS = TransformersType.LOG
_C.TRANSFORMATION.NCEP_DEW_POINT_TEMP_C_NO_NANS = TransformersType.LOG
# -----------------------------------------------------------------------------
# DECOMPOSITION
# -----------------------------------------------------------------------------
_C.PCA = CN()
_C.PCA.N_COMPONENTS = 0.9  # number of components
_C.PCA.PLOT = False  # set True if you want to plot pca components

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# _C.OUTPUT_DIR = "../outputs"

# -----------------------------------------------------------------------------
# SAMPLING
# -----------------------------------------------------------------------------
_C.RANDOM_UNDER_SAMPLER = CN()
_C.RANDOM_UNDER_SAMPLER.SAMPLING_STRATEGY = 'auto'  # float, str, dict, callable, default=’auto’
_C.RANDOM_UNDER_SAMPLER.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.RANDOM_UNDER_SAMPLER.REPLACEMENT = False  # bool, default=False

_C.RANDOM_OVER_SAMPLER = CN()
_C.RANDOM_OVER_SAMPLER.SAMPLING_STRATEGY = 'minority'  # float, str, dict or callable, default=’auto’
_C.RANDOM_OVER_SAMPLER.RANDOM_STATE = 2021  # int, RandomState instance, default=None
# _C.RANDOM_OVER_SAMPLER.SHRINKAGE = 0  # float or dict, default=None

_C.SMOTE = CN()
_C.SMOTE.SAMPLING_STRATEGY = 'auto'  # float, str, dict or callable, default=’auto’ {'minority'}
_C.SMOTE.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SMOTE.K_NEIGHBORS = 5  # int or object, default=5
_C.SMOTE.N_JOBS = -1  # int, default=None

_C.SMOTENC = CN()
_C.SMOTENC.CATEGORICAL_FEATURES = ('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                                   'month')  # ndarray of shape (n_cat_features,) or (n_features,)
_C.SMOTENC.SAMPLING_STRATEGY = 'minority'  # float, str, dict or callable, default=’auto’
_C.SMOTENC.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SMOTENC.K_NEIGHBORS = 5  # int or object, default=5
_C.SMOTENC.N_JOBS = -1  # int, default=None

_C.SVMSMOTE = CN()
_C.SVMSMOTE.SAMPLING_STRATEGY = 'auto'  # float, str, dict or callable, default=’auto’ {'minority'}
_C.SVMSMOTE.RANDOM_STATE = 2021  # int, RandomState instance, default=None
_C.SVMSMOTE.K_NEIGHBORS = 3  # int or object, default=5
_C.SVMSMOTE.N_JOBS = -1  # int, default=None
_C.SVMSMOTE.M_NEIGHBORS = 10  # int or object, default=10
# _C.SVMSMOTE.SVM_ESTIMATOR = 5  # estimator object, default=SVC()
_C.SVMSMOTE.OUT_STEP = 0.5  # float, default=0.5

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
# support vector machine for classification task

_C.SVM = CN()
_C.SVM.NAME = 'SVM'

_C.SVM.C = 10  # float, default=1.0
_C.SVM.KERNEL = 'rbf'  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
_C.SVM.DEGREE = 1  # int, default=3
_C.SVM.GAMMA = 'scale'  # {'scale', 'auto'} or float, default='scale'
_C.SVM.COEF0 = 0.0  # float, default=0.0
_C.SVM.SHRINKING = True  # bool, default=True
_C.SVM.PROBABILITY = False  # bool, default=False
_C.SVM.TOL = 1e-3  # float, default=1e-3
_C.SVM.CACHE_SIZE = 200  # float, default=200
_C.SVM.CLASS_WEIGHT = None  # dict or 'balanced', default=None
_C.SVM.VERBOSE = True  # bool, default=False
_C.SVM.MAX_ITER = -1  # int, default=-1
_C.SVM.DECISION_FUNCTION_SHAPE = 'ovr'  # {'ovo', 'ovr'}, default='ovr'
_C.SVM.BREAK_TIES = False  # bool, default=False
_C.SVM.RANDOM_STATE = _C.BASIC.RAND_STATE  # int or RandomState instance, default=None

# support vector machine for regression task
_C.SVR = CN()
_C.SVR.NAME = 'SVM'

_C.SVR.KERNEL = 'poly'  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
_C.SVR.DEGREE = 3  # int, default=3
_C.SVR.GAMMA = 'auto'  # {'scale', 'auto'} or float, default='scale'
_C.SVR.COEF0 = 0.0  # float, default=0.0
_C.SVR.TOL = 1e-3  # float, default=1e-3
_C.SVR.C = 1.0  # float, default=1.0
_C.SVR.EPSILON = 0.1  # float, default=0.1
_C.SVR.SHRINKING = True  # bool, default=True
_C.SVR.CACHE_SIZE = 200  # float, default=200
_C.SVR.VERBOSE = True  # bool, default=False
_C.SVR.MAX_ITER = -1  # int, default=-1

# configurations for hyperparameter tuning
_C.SVM.HYPER_PARAM_TUNING = CN()
_C.SVM.HYPER_PARAM_TUNING.KERNEL = ('linear', 'poly', 'rbf')
_C.SVM.HYPER_PARAM_TUNING.C = (0.1, 1, 10)
_C.SVM.HYPER_PARAM_TUNING.DEGREE = (1, 2, 3)
_C.SVM.HYPER_PARAM_TUNING.GAMMA = ('scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001)
_C.SVM.HYPER_PARAM_TUNING.COEF0 = None
_C.SVM.HYPER_PARAM_TUNING.SHRINKING = None
_C.SVM.HYPER_PARAM_TUNING.PROBABILITY = None
_C.SVM.HYPER_PARAM_TUNING.TOL = None
_C.SVM.HYPER_PARAM_TUNING.CACHE_SIZE = None
_C.SVM.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.SVM.HYPER_PARAM_TUNING.MAX_ITER = None
_C.SVM.HYPER_PARAM_TUNING.DECISION_FUNCTION_SHAPE = None
_C.SVM.HYPER_PARAM_TUNING.BREAK_TIES = None

_C.SVR.HYPER_PARAM_TUNING = CN()
_C.SVR.HYPER_PARAM_TUNING.KERNEL = None
_C.SVR.HYPER_PARAM_TUNING.DEGREE = None
_C.SVR.HYPER_PARAM_TUNING.GAMMA = None
_C.SVR.HYPER_PARAM_TUNING.COEF0 = None
_C.SVR.HYPER_PARAM_TUNING.TOL = None
_C.SVR.HYPER_PARAM_TUNING.C = None
_C.SVR.HYPER_PARAM_TUNING.EPSILON = None
_C.SVR.HYPER_PARAM_TUNING.SHRINKING = None
_C.SVR.HYPER_PARAM_TUNING.CACHE_SIZE = None
_C.SVR.HYPER_PARAM_TUNING.VERBOSE = None
_C.SVR.HYPER_PARAM_TUNING.MAX_ITER = None
# ----------------------------------------------------------------------------
_C.RANDOM_FOREST = CN()
_C.RANDOM_FOREST.NAME = 'RANDOM_FOREST'

_C.RANDOM_FOREST.N_ESTIMATORS = 1000  # int, default=100
_C.RANDOM_FOREST.CRITERION = "mse"  # classification:{"gini", "entropy"}, default="gini" | regression:{"mse", "mae"}, default="mse"
_C.RANDOM_FOREST.MAX_DEPTH = 10  # int, default=None
_C.RANDOM_FOREST.MIN_SAMPLES_SPLIT = 5  # int or float, default=2
_C.RANDOM_FOREST.MIN_SAMPLES_LEAF = 2  # int or float, default=1
_C.RANDOM_FOREST.MIN_WEIGHT_FRACTION_LEAF = 0.0  # float, default=0.0
_C.RANDOM_FOREST.MAX_FEATURES = "auto"  # {"auto", "sqrt", "log2"}, int or float, default="auto"
_C.RANDOM_FOREST.MAX_LEAF_NODES = None  # int, default=None
_C.RANDOM_FOREST.MIN_IMPURITY_DECREASE = 0.0  # float, default=0.0
_C.RANDOM_FOREST.MIN_IMPURITY_SPLIT = None  # float, default=None
_C.RANDOM_FOREST.BOOTSTRAP = True  # bool, default=True
_C.RANDOM_FOREST.OOB_SCORE = False  # bool, default=False
_C.RANDOM_FOREST.N_JOBS = None  # int, default=None
_C.RANDOM_FOREST.RANDOM_STATE = _C.BASIC.RAND_STATE  # int or RandomState, default=None
_C.RANDOM_FOREST.VERBOSE = 0  # int, default=0
_C.RANDOM_FOREST.WARM_START = False  # bool, default=False
_C.RANDOM_FOREST.CLASS_WEIGHT = None  # "balanced", "balanced_subsample"}, dict or list of dicts, default=None
_C.RANDOM_FOREST.CCP_ALPHA = 0.0  # non-negative float, default=0.0
_C.RANDOM_FOREST.MAX_SAMPLES = None  # int or float, default=None

# configurations for hyperparameter tuning
_C.RANDOM_FOREST.HYPER_PARAM_TUNING = CN()
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.N_ESTIMATORS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.CRITERION = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_DEPTH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_SAMPLES_SPLIT = [2, 5, 10]
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_SAMPLES_LEAF = [1, 2, 4]
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_WEIGHT_FRACTION_LEAF = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_FEATURES = ['auto', 'sqrt']
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_LEAF_NODES = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_IMPURITY_DECREASE = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MIN_IMPURITY_SPLIT = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.BOOTSTRAP = [True, False]
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.OOB_SCORE = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.WARM_START = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.CCP_ALPHA = None
_C.RANDOM_FOREST.HYPER_PARAM_TUNING.MAX_SAMPLES = None

# ----------------------------------------------------------------------------
_C.LOGISTIC_REGRESSION = CN()
_C.LOGISTIC_REGRESSION.NAME = 'LOGISTIC REGRESSION'

_C.LOGISTIC_REGRESSION.PENALTY = 'l2'  # {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
_C.LOGISTIC_REGRESSION.DUAL = False  # bool, default=False
_C.LOGISTIC_REGRESSION.TOL = 1e-4  # float, default=1e-4
_C.LOGISTIC_REGRESSION.C = 0.1  # float, default=1.0
_C.LOGISTIC_REGRESSION.FIT_INTERCEPT = True  # bool, default=True
_C.LOGISTIC_REGRESSION.INTERCEPT_SCALING = 1  # float, default=1
_C.LOGISTIC_REGRESSION.CLASS_WEIGHT = None  # dict or 'balanced', default=None
_C.LOGISTIC_REGRESSION.RANDOM_STATE = _C.BASIC.RAND_STATE  # int, RandomState instance, default=None
_C.LOGISTIC_REGRESSION.SOLVER = 'liblinear'  # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
_C.LOGISTIC_REGRESSION.MAX_ITER = 100  # int, default=100
_C.LOGISTIC_REGRESSION.MULTI_CLASS = 'auto'  # {'auto', 'ovr', 'multinomial'}, default='auto'
_C.LOGISTIC_REGRESSION.VERBOSE = 0  # int, default=0
_C.LOGISTIC_REGRESSION.WARM_START = False  # bool, default=False
_C.LOGISTIC_REGRESSION.N_JOBS = None  # int, default=None
_C.LOGISTIC_REGRESSION.L1_RATIO = None  # float, default=None

# configurations for hyperparameter tuning
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING = CN()
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.PENALTY = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.DUAL = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.TOL = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.FIT_INTERCEPT = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.INTERCEPT_SCALING = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.SOLVER = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.MAX_ITER = list(range(100, 800, 100))
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.MULTI_CLASS = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.WARM_START = None
_C.LOGISTIC_REGRESSION.HYPER_PARAM_TUNING.L1_RATIO = None

# ----------------------------------------------------------------------------
_C.DECISION_TREE = CN()
_C.DECISION_TREE.NAME = 'DECISION TREE'

_C.DECISION_TREE.CRITERION = "mse"  # criterion : {"gini", "entropy"}, default="gini" | {"mse", "friedman_mse", "mae"}, default="mse"
_C.DECISION_TREE.SPLITTER = "best"  # splitter : {"best", "random"}, default="best"
_C.DECISION_TREE.MAX_DEPTH = 10  # max_depth : int, default=None
_C.DECISION_TREE.MIN_SAMPLES_SPLIT = 10  # min_samples_split : int or float, default=2
_C.DECISION_TREE.MIN_SAMPLES_LEAF = 1  # min_samples_leaf : int or float, default=1
_C.DECISION_TREE.MIN_WEIGHT_FRACTION_LEAF = 0.0  # min_weight_fraction_leaf : float, default=0.0
_C.DECISION_TREE.MAX_FEATURES = 'auto'  # max_features : int, float or {"auto", "sqrt", "log2"}, default=None
_C.DECISION_TREE.RANDOM_STATE = _C.BASIC.RAND_STATE  # random_state : int, RandomState instance, default=None
_C.DECISION_TREE.MAX_LEAF_NODES = None  # max_leaf_nodes : int, default=None
_C.DECISION_TREE.MIN_IMPURITY_DECREASE = 0.0  # min_impurity_decrease : float, default=0.0
_C.DECISION_TREE.MIN_IMPURITY_SPLIT = None  # min_impurity_split : float, default=0
_C.DECISION_TREE.CLASS_WEIGHT = None  # class_weight : dict, list of dict or "balanced", default=None
_C.DECISION_TREE.PRESORT = 'deprecated'  # presort : deprecated, default='deprecated'
_C.DECISION_TREE.CCP_ALPHA = 0.0  # ccp_alpha : non-negative float, default=0.0

# configurations for hyperparameter tuning
_C.DECISION_TREE.HYPER_PARAM_TUNING = CN()
_C.DECISION_TREE.HYPER_PARAM_TUNING.CRITERION = ("gini", "entropy")
_C.DECISION_TREE.HYPER_PARAM_TUNING.SPLITTER = ("best", "random")
_C.DECISION_TREE.HYPER_PARAM_TUNING.MAX_DEPTH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_SAMPLES_SPLIT = [2, 5, 10]
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_SAMPLES_LEAF = [1, 2, 4]
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_WEIGHT_FRACTION_LEAF = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MAX_FEATURES = ['auto', 'sqrt']
_C.DECISION_TREE.HYPER_PARAM_TUNING.MAX_LEAF_NODES = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_IMPURITY_DECREASE = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.MIN_IMPURITY_SPLIT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.CLASS_WEIGHT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.PRESORT = None
_C.DECISION_TREE.HYPER_PARAM_TUNING.CCP_ALPHA = None
# ----------------------------------------------------------------------------
_C.KNNC = CN()
_C.KNNC.NAME = 'KNNClassifier'
_C.KNNC.N_NEIGHBORS = 5  # n_neighbors : int, default=5 Number of neighbors to use by default for :meth:`kneighbors` queries.
_C.KNNC.WEIGHTS = 'uniform'  # weights : {'uniform', 'distance'} or callable, default='uniform'
_C.KNNC.ALGORITHM = 'auto'  # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
_C.KNNC.LEAF_SIZE = 30  # leaf_size : int, default=30
_C.KNNC.P = 2  # p : int, default=2 Power parameter for the Minkowski metric.
_C.KNNC.METRIC = 'minkowski'  # metric : str or callable, default='minkowski'
_C.KNNC.METRIC_PARAMS = None  # metric_params : dict, default=None
_C.KNNC.N_JOBS = -1  # n_jobs : int, default=None

_C.KNNC.HYPER_PARAM_TUNING = CN()
_C.KNNC.HYPER_PARAM_TUNING.N_NEIGHBORS = None
_C.KNNC.HYPER_PARAM_TUNING.WEIGHTS = None
_C.KNNC.HYPER_PARAM_TUNING.ALGORITHM = None
_C.KNNC.HYPER_PARAM_TUNING.LEAF_SIZE = None
_C.KNNC.HYPER_PARAM_TUNING.P = None
_C.KNNC.HYPER_PARAM_TUNING.METRIC = None
_C.KNNC.HYPER_PARAM_TUNING.METRIC_PARAMS = None
_C.KNNC.HYPER_PARAM_TUNING.N_JOBS = None

_C.KNNR = CN()
_C.KNNR.NAME = 'KNNClassifier'
_C.KNNR.N_NEIGHBORS = 7  # n_neighbors : int, default=5 Number of neighbors to use by default for :meth:`kneighbors` queries.
_C.KNNR.WEIGHTS = 'distance'  # weights : {'uniform', 'distance'} or callable, default='uniform'
_C.KNNR.ALGORITHM = 'auto'  # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
_C.KNNR.LEAF_SIZE = 10  # leaf_size : int, default=30
_C.KNNR.P = 1  # p : int, default=2 Power parameter for the Minkowski metric.
_C.KNNR.METRIC = 'minkowski'  # metric : str or callable, default='minkowski'
_C.KNNR.METRIC_PARAMS = None  # metric_params : dict, default=None
_C.KNNR.N_JOBS = -1  # n_jobs : int, default=None

_C.KNNR.HYPER_PARAM_TUNING = CN()
_C.KNNR.HYPER_PARAM_TUNING.N_NEIGHBORS = [2,3,4,5,6,7,8,9,10]
_C.KNNR.HYPER_PARAM_TUNING.WEIGHTS = ['distance','uniform']
_C.KNNR.HYPER_PARAM_TUNING.ALGORITHM = ['auto', 'ball_tree', 'kd_tree', 'brute']
_C.KNNR.HYPER_PARAM_TUNING.LEAF_SIZE = [10,20,30,50,70,100]
_C.KNNR.HYPER_PARAM_TUNING.P = [1,2,3]
_C.KNNR.HYPER_PARAM_TUNING.METRIC = None
_C.KNNR.HYPER_PARAM_TUNING.METRIC_PARAMS = None
_C.KNNR.HYPER_PARAM_TUNING.N_JOBS = None

# ----------------------------------------------------------------------------
_C.MLP = CN()
_C.MLP.NAME = 'MLP'
_C.MLP.HIDDEN_LAYER_SIZES = (500,)  # hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)
_C.MLP.ACTIVATION = 'relu'  # activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
_C.MLP.SOLVER = 'sgd'  # solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
_C.MLP.ALPHA = 0.009  # alpha : float, default=0.0001
_C.MLP.BATCH_SIZE = 'auto'  # batch_size : int, default='auto'
_C.MLP.LEARNING_RATE = 'constant'  # learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
_C.MLP.LEARNING_RATE_INIT = 0.01  # learning_rate_init : double, default=0.001
_C.MLP.POWER_T = 0.5  # power_t : double, default=0.5
_C.MLP.MAX_ITER = 2000  # max_iter : int, default=200
_C.MLP.SHUFFLE = True  # shuffle : bool, default=True
_C.MLP.RANDOM_STATE = _C.BASIC.RAND_STATE  # random_state : int, RandomState instance, default=None
_C.MLP.TOL = 1e-4  # tol : float, default=1e-4
_C.MLP.VERBOSE = True  # verbose : bool, default=False
_C.MLP.WARM_START = False  # warm_start : bool, default=False
_C.MLP.MOMENTUM = 0.9  # momentum : float, default=0.9
_C.MLP.NESTEROVS_MOMENTUM = True  # nesterovs_momentum : boolean, default=True
_C.MLP.EARLY_STOPPING = False  # early_stopping : bool, default=False
_C.MLP.VALIDATION_FRACTION = 0.1  # validation_fraction : float, default=0.1
_C.MLP.BETA_1 = 0.9  # beta_1 : float, default=0.9
_C.MLP.BETA_2 = 0.999  # beta_2 : float, default=0.999
_C.MLP.EPSILON = 1e-8  # epsilon : float, default=1e-8
_C.MLP.N_ITER_NO_CHANGE = 10  # n_iter_no_change : int, default=10
_C.MLP.MAX_FUN = 15000  # max_fun : int, default=15000

_C.MLP.HYPER_PARAM_TUNING = CN()
_C.MLP.HYPER_PARAM_TUNING.NAME = None
_C.MLP.HYPER_PARAM_TUNING.HIDDEN_LAYER_SIZES = None
_C.MLP.HYPER_PARAM_TUNING.ACTIVATION = None
_C.MLP.HYPER_PARAM_TUNING.SOLVER = None
_C.MLP.HYPER_PARAM_TUNING.ALPHA = None
_C.MLP.HYPER_PARAM_TUNING.BATCH_SIZE = None
_C.MLP.HYPER_PARAM_TUNING.LEARNING_RATE = None
_C.MLP.HYPER_PARAM_TUNING.LEARNING_RATE_INIT = None
_C.MLP.HYPER_PARAM_TUNING.POWER_T = None
_C.MLP.HYPER_PARAM_TUNING.MAX_ITER = None
_C.MLP.HYPER_PARAM_TUNING.SHUFFLE = None
_C.MLP.HYPER_PARAM_TUNING.RANDOM_STATE = None
_C.MLP.HYPER_PARAM_TUNING.TOL = None
_C.MLP.HYPER_PARAM_TUNING.VERBOSE = None
_C.MLP.HYPER_PARAM_TUNING.WARM_START = None
_C.MLP.HYPER_PARAM_TUNING.MOMENTUM = None
_C.MLP.HYPER_PARAM_TUNING.NESTEROVS_MOMENTUM = None
_C.MLP.HYPER_PARAM_TUNING.EARLY_STOPPING = None
_C.MLP.HYPER_PARAM_TUNING.VALIDATION_FRACTION = None
_C.MLP.HYPER_PARAM_TUNING.BETA_1 = None
_C.MLP.HYPER_PARAM_TUNING.BETA_2 = None
_C.MLP.HYPER_PARAM_TUNING.EPSILON = None
_C.MLP.HYPER_PARAM_TUNING.N_ITER_NO_CHANGE = None
_C.MLP.HYPER_PARAM_TUNING.MAX_FUN = None

# ----------------------------------------------------------------------------
_C.XGBOOST = CN()
_C.XGBOOST.NAME = 'xgboost'
_C.XGBOOST.N_ESTIMATORS = 100  # Number of gradient boosted trees. Equivalent to number of boosting rounds.
_C.XGBOOST.MAX_DEPTH = 50  # max_depth : Maximum tree depth for base learners.
_C.XGBOOST.LEARNING_RATE = 0.07  # learning_rate : Boosting learning rate (xgb’s “eta”)
_C.XGBOOST.VERBOSITY = 1  # verbosity : The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
_C.XGBOOST.OBJECTIVE = None  # objective : Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).
_C.XGBOOST.BOOSTER = None  # booster: Specify which booster to use: gbtree, gblinear or dart.
_C.XGBOOST.TREE_METHOD = None  # tree_method:  Specify which tree method to use.  Default to auto.  If this parameter is set to default, XGBoost will choose the most conservative option available.  It's recommended to study this option from parameters document.
_C.XGBOOST.N_JOBS = -1  # n_jobs : Number of parallel threads used to run xgboost.
_C.XGBOOST.GAMMA = None  # gamma : Minimum loss reduction required to make a further partition on a leaf node of the tree.
_C.XGBOOST.MIN_CHILD_WEIGHT = None  # min_child_weight : Minimum sum of instance weight(hessian) needed in a child.
_C.XGBOOST.MAX_DELTA_STEP = None  # max_delta_step : Maximum delta step we allow each tree's weight estimation to be.
_C.XGBOOST.SUBSAMPLE = None  # subsample : Subsample ratio of the training instance.
_C.XGBOOST.COLSAMPLE_BYTREE = None  # colsample_bytree : Subsample ratio of columns when constructing each tree.
_C.XGBOOST.COLSAMPLE_BYLEVEL = None  # colsample_bylevel : Subsample ratio of columns for each level.
_C.XGBOOST.COLSAMPLE_BYNODE = None  # colsample_bynode :   Subsample ratio of columns for each split.
_C.XGBOOST.REG_ALPHA = None  # reg_alpha : L1 regularization term on weights
_C.XGBOOST.REG_LAMBDA = 0.9  # reg_lambda : L2 regularization term on weights
_C.XGBOOST.SCALE_POS_WEIGHT = None  # scale_pos_weight : Balancing of positive and negative weights.
_C.XGBOOST.BASE_SCORE = None  # base_score: The initial prediction score of all instances, global bias.
_C.XGBOOST.RANDOM_STATE = _C.BASIC.RAND_STATE  # random_state : Random number seed.
_C.XGBOOST.MISSING = None  # missing : Value in the data which needs to be present as a missing value.  float, default np.nan
_C.XGBOOST.NUM_PARALLEL_TREE = None  # num_parallel_tree: Used for boosting random forest.
_C.XGBOOST.MONOTONE_CONSTRAINTS = None  # monotone_constraints : Constraint of variable monotonicity.
_C.XGBOOST.INTERACTION_CONSTRAINTS = None  # interaction_constraints :  Constraints for interaction representing permitted interactions.
_C.XGBOOST.IMPORTANCE_TYPE = "gain"  # importance_type:  The feature importance type for the feature_importances either "gain", "weight", "cover", "total_gain" or "total_cover".
_C.XGBOOST.GPU_ID = None  #
_C.XGBOOST.VALIDATE_PARAMETERS = None  #

_C.XGBOOST.HYPER_PARAM_TUNING = CN()
_C.XGBOOST.HYPER_PARAM_TUNING.N_ESTIMATORS = None
_C.XGBOOST.HYPER_PARAM_TUNING.MAX_DEPTH = None
_C.XGBOOST.HYPER_PARAM_TUNING.LEARNING_RATE = None
_C.XGBOOST.HYPER_PARAM_TUNING.VERBOSITY = None
_C.XGBOOST.HYPER_PARAM_TUNING.OBJECTIVE = None
_C.XGBOOST.HYPER_PARAM_TUNING.BOOSTER = None
_C.XGBOOST.HYPER_PARAM_TUNING.TREE_METHOD = None
_C.XGBOOST.HYPER_PARAM_TUNING.N_JOBS = None
_C.XGBOOST.HYPER_PARAM_TUNING.GAMMA = None
_C.XGBOOST.HYPER_PARAM_TUNING.MIN_CHILD_WEIGHT = None
_C.XGBOOST.HYPER_PARAM_TUNING.MAX_DELTA_STEP = None
_C.XGBOOST.HYPER_PARAM_TUNING.SUBSAMPLE = None
_C.XGBOOST.HYPER_PARAM_TUNING.COLSAMPLE_BYTREE = None
_C.XGBOOST.HYPER_PARAM_TUNING.COLSAMPLE_BYLEVEL = None
_C.XGBOOST.HYPER_PARAM_TUNING.COLSAMPLE_BYNODE = None
_C.XGBOOST.HYPER_PARAM_TUNING.REG_ALPHA = None
_C.XGBOOST.HYPER_PARAM_TUNING.REG_LAMBDA = None
_C.XGBOOST.HYPER_PARAM_TUNING.SCALE_POS_WEIGHT = None
_C.XGBOOST.HYPER_PARAM_TUNING.BASE_SCORE = None
_C.XGBOOST.HYPER_PARAM_TUNING.MISSING = None
_C.XGBOOST.HYPER_PARAM_TUNING.NUM_PARALLEL_TREE = None
_C.XGBOOST.HYPER_PARAM_TUNING.MONOTONE_CONSTRAINTS = None
_C.XGBOOST.HYPER_PARAM_TUNING.INTERACTION_CONSTRAINTS = None
_C.XGBOOST.HYPER_PARAM_TUNING.IMPORTANCE_TYPE = None
_C.XGBOOST.HYPER_PARAM_TUNING.GPU_ID = None
_C.XGBOOST.HYPER_PARAM_TUNING.VALIDATE_PARAMETERS = None
# ----------------------------------------------------------------------------
_C.LIGHTGBM = CN()
_C.LIGHTGBM.NAME = 'LIGHTGBM'
_C.LIGHTGBM.BOOSTING_TYPE = 'gbdt'  # boosting_type : string, optional (default='gbdt') dart goss rf
_C.LIGHTGBM.NUM_LEAVES = 50  # num_leaves : int, optional (default=31)
_C.LIGHTGBM.MAX_DEPTH = -1  # max_depth : int, optional (default=-1)
_C.LIGHTGBM.LEARNING_RATE = 0.1  # learning_rate : float, optional (default=0.1)
_C.LIGHTGBM.N_ESTIMATORS = 1500  # n_estimators : int, optional (default=100)
_C.LIGHTGBM.RANDOM_STATE = _C.BASIC.RAND_STATE

_C.LIGHTGBM.HYPER_PARAM_TUNING = CN()
_C.LIGHTGBM.HYPER_PARAM_TUNING.BOOSTING_TYPE = None
_C.LIGHTGBM.HYPER_PARAM_TUNING.NUM_LEAVES = None
_C.LIGHTGBM.HYPER_PARAM_TUNING.MAX_DEPTH = None
_C.LIGHTGBM.HYPER_PARAM_TUNING.LEARNING_RATE = None
_C.LIGHTGBM.HYPER_PARAM_TUNING.N_ESTIMATORS = None
# ----------------------------------------------------------------------------
_C.CATBOOST = CN()
_C.CATBOOST.NAME = 'catboost'
_C.CATBOOST.ITERATIONS = 500  # iterations : int, [default=500]
_C.CATBOOST.LEARNING_RATE = 0.1  # learning_rate : float,
_C.CATBOOST.DEPTH = 16  # depth : int, [default=6] Depth of a tree. All trees are the same depth.
_C.CATBOOST.L2_LEAF_REG = 3.0  # l2_leaf_reg : float, [default=3.0] Coefficient at the L2 regularization term of the cost function.
_C.CATBOOST.LOSS_FUNCTION = 'MAE'  # loss_function : string, [default='RMSE'] 'RMSE' 'MAE' 'Quantile:alpha=value' 'LogLinQuantile:alpha=value' 'Poisson' 'MAPE' 'Lq:q=value' The metric to use in training and also selector of the machine learning  [default='Logloss']
_C.CATBOOST.RANDOM_SEED = _C.BASIC.RAND_STATE  # random_seed : int, [default=None] Random number seed.
_C.CATBOOST.USE_BEST_MODEL = False  # use_best_model : bool, [default=None] To limit the number of trees in predict() using information about the optimal value of the error function.
_C.CATBOOST.VERBOSE = True  # verbose: bool
_C.CATBOOST.EVAL_METRIC = None  # eval_metric : string or object, [default=None] To optimize your custom metric in loss.
_C.CATBOOST.BOOSTING_TYPE = 'Plain'  # boosting_type : string, default value depends on object count and feature count in train dataset and on learning mode. - 'Ordered' - Gives better quality, but may slow down the training.  - 'Plain' - The classic gradient boosting scheme. May result in quality degradation, but does not slow down the training.
_C.CATBOOST.TASK_TYPE = 'CPU'  # task_type : string, [default=None]  - 'CPU' - 'GPU'
_C.CATBOOST.N_ESTIMATORS = 500  # n_estimators : int, synonym for iterations.
_C.CATBOOST.CAT_FEATURES = None  # cat_features : list or numpy.ndarray, [default=None]

_C.CATBOOST.HYPER_PARAM_TUNING = CN()
_C.CATBOOST.HYPER_PARAM_TUNING.ITERATIONS = None
_C.CATBOOST.HYPER_PARAM_TUNING.LEARNING_RATE = None
_C.CATBOOST.HYPER_PARAM_TUNING.DEPTH = None
_C.CATBOOST.HYPER_PARAM_TUNING.L2_LEAF_REG = None
_C.CATBOOST.HYPER_PARAM_TUNING.LOSS_FUNCTION = None
_C.CATBOOST.HYPER_PARAM_TUNING.RANDOM_SEED = None
_C.CATBOOST.HYPER_PARAM_TUNING.USE_BEST_MODEL = None
_C.CATBOOST.HYPER_PARAM_TUNING.VERBOSE = None
_C.CATBOOST.HYPER_PARAM_TUNING.EVAL_METRIC = None
_C.CATBOOST.HYPER_PARAM_TUNING.BOOSTING_TYPE = None
_C.CATBOOST.HYPER_PARAM_TUNING.TASK_TYPE = None
_C.CATBOOST.HYPER_PARAM_TUNING.N_ESTIMATORS = None
_C.CATBOOST.HYPER_PARAM_TUNING.CAT_FEATURES = None
# ----------------------------------------------------------------------------
_C.LR = CN()
_C.LR.NAME = 'LR'
_C.LR.FIT_INTERCEPT = True  #  fit_intercept : bool, default=True
_C.LR.NORMALIZE = False  #  normalize : bool, default=False
_C.LR.COPY_X = True  # copy_X : bool, default=True
_C.LR.N_JOBS = -1  # learning_rate : float,

_C.LR.HYPER_PARAM_TUNING = CN()
_C.LR.HYPER_PARAM_TUNING.FIT_INTERCEPT = None
_C.LR.HYPER_PARAM_TUNING.NORMALIZE = None
_C.LR.HYPER_PARAM_TUNING.COPY_X = None
