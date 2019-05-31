import copy
import errno
import random
import glob
import os.path
import time
import calendar
import json
import pickle
import netCDF4
import numpy
import keras
import datetime 
import scipy.io as sio
from keras import backend as K
import tensorflow
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator)
import matplotlib.colors
import matplotlib.pyplot as pyplot
import sklearn.metrics
import keras_metrics

# Variable names.
MAT_Uin = 'u10_hourly'
MAT_Vin = 'v10_hourly'
# NETCDF_V_WIND_NAME = 'V10_curr'

NETCDF_PREDICTOR_NAMES = [
    MAT_Uin, MAT_Vin
]

NETCDF_TARGET_NAME = 'v10_hourly_out'
TARGET_NAME = 'v10_hourly_out'


PREDICTOR_NAMES = [
    MAT_Uin, MAT_Vin
]

DIR_NAME = '.'


MIN_XENTROPY_DECREASE_FOR_EARLY_STOP = 0.005
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.00001
NUM_EPOCHS_FOR_EARLY_STOPPING = 10
NUM_EPOCHS_FOR_REDUCE_LR = 5
FACTOR_REDUCE_LR = 0.7
MIN_LR_REDUCE_TO = 0.00005
MIN_MSE_DECREASE_FOR_REDUCE_LR = 0.0001

PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'

TRAINING_FILES_KEY = 'training_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
TARGET_DICT_KEY = 'target_dict'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_file_names'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
CNN_FILE_KEY = 'cnn_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'

# Machine-learning constants.
L1_WEIGHT = 0.
L2_WEIGHT = 0.001
NUM_PREDICTORS_TO_FIRST_NUM_FILTERS = 2
NUM_CONV_LAYER_SETS = 2
NUM_CONV_LAYERS_PER_SET = 2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3
CONV_LAYER_DROPOUT_FRACTION = None
USE_BATCH_NORMALIZATION = True
SLOPE_FOR_RELU = 0.2
NUM_POOLING_ROWS = 2
NUM_POOLING_COLUMNS = 2
NUM_DENSE_LAYERS = 3
DENSE_LAYER_DROPOUT_FRACTION = 0.5

NUM_SMOOTHING_FILTER_ROWS = 5
NUM_SMOOTHING_FILTER_COLUMNS = 5


LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

METRIC_FUNCTION_DICT = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_peirce_score': keras_metrics.binary_peirce_score,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}


def datenum_mat_2_python_datetime(datenum_vec):

    Pydater = datetime.fromordinal(int(datenum_vec)) + timedelta(days=datenum_vec%1) - timedelta(days = 366)

    return Pydater 


def read_image_file(netcdf_file_name):
    """Reads storm-centered images from MAT file.
    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    :param netcdf_file_name: Path to input file.
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """


    dataset_object = sio.loadmat(netcdf_file_name)


    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object[this_predictor_name][:], dtype=float
        )
        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1)

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )

    target_matrix = numpy.array(
        dataset_object[NETCDF_TARGET_NAME][:], dtype=float
    )

    return {
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix
    }

def read_Qscat_file(netcdf_file_name):
    """Reads storm-centered images from MAT file.
    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    :param netcdf_file_name: Path to input file.
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """


    dataset_object = sio.loadmat(netcdf_file_name)


    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object[this_predictor_name][:], dtype=float
        )
        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1)

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )

    return {
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
    }



def normalize_images(predictor_matrix, predictor_names, normalization_dict=None):
    """Normalizes images to z-scores.
    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = numpy.mean(predictor_matrix[..., m])
            this_stdev = numpy.std(predictor_matrix[..., m], ddof=1)

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_mean, this_stdev]
            )

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            (predictor_matrix[..., m] - this_mean) / float(this_stdev)
        )

    return predictor_matrix, normalization_dict

def min_max_scale(X, range=(0, 1)):
    mi, ma = range
    Xmin = X.min()
    Xmax = X.max()
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (ma - mi) + mi
    return X_scaled, Xmin, Xmax 

def MinMax_All_images(file_path_list, predictor_names, normalization_dict=None):
    """Normalizes images between 0 and 1.
    E = number of examples in file
    M = number of columns in each grid (lon)
    N = number of rows in grid (lat)
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """
    num_predictors = len(predictor_names)
    print(num_predictors)
    num_files = len(file_path_list)
    print(num_files)

    FileIN = read_image_file(file_path_list[0])
    predictor_mat = FileIN['predictor_matrix']

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_max = numpy.amax(predictor_mat[..., m])
            this_min = numpy.amin(predictor_mat[..., m])

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_max, this_min]
            )
    print('dict:', normalization_dict)

    for ii in range(num_files):
        print(file_path_list[ii])
        FileIN = read_image_file(file_path_list[ii])
        predictor_mat = FileIN['predictor_matrix']

        for m in range(num_predictors):
            this_max = numpy.amax(predictor_mat[..., m])
            this_min = numpy.amin(predictor_mat[..., m])
            print('m,max,min:',m,this_max,this_min)

            if this_max >  normalization_dict[predictor_names[m]][0]:
                normalization_dict[predictor_names[m]] = numpy.array(
                    [this_max, normalization_dict[predictor_names[m]][1]]
                )
                print('dict:', normalization_dict)
                print('update max:',this_max)

            if this_min <  normalization_dict[predictor_names[m]][1]:
                normalization_dict[predictor_names[m]] = numpy.array(
                    [normalization_dict[predictor_names[m]][0], this_min]
                )
                print('dict:', normalization_dict)
                print('update min:', this_min)
        del FileIN, predictor_mat
    return normalization_dict



def MinMax_norm(predictor_matrix, predictor_names,range_high,range_low,normalization_dict):
    """Normalizes images between 0 and 1 by some global min and max.
    E = number of examples (storm objects) in file
    M = number of rows in each grid (lon)
    N = number of columns in grid (lat)
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-N-by-M-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [min, max] to normalize the input. 
    :param range_low = normalization range low 
    :param range_high = normalization range high

    :return: predictor_matrix: Normalized version of input.
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names)

    for m in range(num_predictors):
        this_max = normalization_dict[predictor_names[m]][0]
        this_min = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            ((predictor_matrix[..., m] - this_min) / (this_max-this_min)) * (range_high-range_low) + range_low
        )

    return predictor_matrix, normalization_dict


def MinMax_norm_targ(predictor_matrix, predictor_names,range_high,range_low,normalization_dict):
    """Normalizes images between 0 and 1 by some global min and max, for target variable. ONLY 1 possible..
    E = number of examples (storm objects) in file
    M = number of rows in each grid (lon)
    N = number of columns in grid (lat)
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-N-by-M-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [min, max] to normalize the input. 
    :param range_low = normalization range low 
    :param range_high = normalization range high

    :return: predictor_matrix: Normalized version of input.
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names.split())

    for m in range(num_predictors):
        this_max = normalization_dict[predictor_names.split()[0]][0]
        this_min = normalization_dict[predictor_names.split()[0]][1]

        predictor_matrix = (
            ((predictor_matrix - this_min) / (this_max-this_min)) * (range_high-range_low) + range_low
        )

    return predictor_matrix, normalization_dict



def rescale_MinMax_norm(predictor_matrix, predictor_names,range_high,range_low,normalization_dict):
    """Normalizes images between 0 and 1 by some global min and max.
    E = number of examples (storm objects) in file
    M = number of rows in each grid (lon)
    N = number of columns in grid (lat)
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-N-by-M-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [min, max] to normalize the input. 
    :param range_low = normalization range low 
    :param range_high = normalization range high

    :return: predictor_matrix: Normalized version of input.
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names)

    for m in range(num_predictors):
        this_max = normalization_dict[predictor_names[m]][0]
        this_min = normalization_dict[predictor_names[m]][1]


        predictor_matrix[..., m] = (((predictor_matrix[...,m]-range_low)/(range_high-range_low))*(this_max-this_min))+this_min
            

    return predictor_matrix, normalization_dict


def rescale_MinMax_norm_targ(predictor_matrix, predictor_names,range_high,range_low,normalization_dict):
    """Normalizes images between 0 and 1 by some global min and max, for target variable. ONLY 1 possible..
    E = number of examples (storm objects) in file
    M = number of rows in each grid (lon)
    N = number of columns in grid (lat)
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-N-by-M-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [min, max] to normalize the input. 
    :param range_low = normalization range low 
    :param range_high = normalization range high

    :return: predictor_matrix: Normalized version of input.
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names.split())

    for m in range(num_predictors):
        this_max = normalization_dict[predictor_names.split()[0]][0]
        this_min = normalization_dict[predictor_names.split()[0]][1]

        predictor_matrix = (
            (((predictor_matrix-range_low)/(range_high-range_low))*(this_max-this_min))+this_min
        )

    return predictor_matrix, normalization_dict



def MinMax_norm_images(predictor_matrix, predictor_names,range_low,range_high,normalization_dict=None):
    """Normalizes images between 0 and 1.
    E = number of examples (storm objects) in file
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """

    num_predictors = len(predictor_names)

    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_max = numpy.amax(predictor_matrix[..., m])
            this_min = numpy.amin(predictor_matrix[..., m])

            normalization_dict[predictor_names[m]] = numpy.array(
                [this_max, this_min]
            )

    for m in range(num_predictors):
        this_max = normalization_dict[predictor_names[m]][0]
        this_min = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            ((predictor_matrix[..., m] - this_min) / (this_max-this_min)) * (range_high-range_low) + range_low
        )

    return predictor_matrix, normalization_dict




def deep_learning_generator(netcdf_file_names, num_examples_per_batch,
                            range_high,range_low,normalization_dict,
                            normalization_dict_target):
    """Generates training examples for deep-learning model on the fly.
    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    C = number of channels (predictor variables)
    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None. Use Min_Max_All Dictionary. 
    :param range_high: normalization range high 
    :param range_low: normalization range low 

    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    # TODO(thunderhoser): Maybe add upsampling or downsampling.

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    predictor_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]
            ))

            this_image_dict = read_image_file(netcdf_file_names[file_index])
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            target_names = this_image_dict[TARGET_NAME_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.

            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0
                )

                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                    axis=0
                )

            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        predictor_matrix, _ = MinMax_norm(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict,
            range_low=range_low,
            range_high=range_high)
        predictor_matrix = predictor_matrix.astype('float32')

        target_values = MinMax_norm_targ(
            predictor_matrix=full_target_matrix[batch_indices, ...],
            predictor_names=target_names,
            normalization_dict=normalization_dict_target,
            range_low=range_low,
            range_high=range_high)
        target_values = target_values[0] #just the predictor values. 

        # target_values = MinMax_norm_targ_mat(
        #     predictor_matrix=full_target_matrix[batch_indices, ...],
        #     predictor_names=target_names.split(),
        #     normalization_dict=normalization_dict_target,
        #     range_low=range_low,
        #     range_high=range_high)
        # target_values = target_values[0] #just the predictor values. 

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix, target_values)


def _create_directory(directory_name=None, file_name=None):
    """Creates directory (along with parents if necessary).
    This method creates directories only when necessary, so you don't have to
    worry about it overwriting anything.
    :param directory_name: Name of desired directory.
    :param file_name: [used only if `directory_name is None`]
        Path to desired file.  All directories in path will be created.
    """

    if directory_name is None:
        directory_name = os.path.split(file_name)[0]

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise


def _get_dense_layer_dimensions(num_input_units, num_classes, num_dense_layers):
    """Returns dimensions (number of input and output units) for each dense lyr.
    D = number of dense layers
    :param num_input_units: Number of input units (features created by
        flattening layer).
    :param num_classes: Number of output classes (possible values of target
        variable).
    :param num_dense_layers: Number of dense layers.
    :return: num_inputs_by_layer: length-D numpy array with number of input
        units by dense layer.
    :return: num_outputs_by_layer: length-D numpy array with number of output
        units by dense layer.
    """

    if num_classes == 2:
        num_output_units = 1
    else:
        num_output_units = num_classes + 0

    e_folding_param = (
        float(-1 * num_dense_layers) /
        numpy.log(float(num_output_units) / num_input_units)
    )

    dense_layer_indices = numpy.linspace(
        0, num_dense_layers - 1, num=num_dense_layers, dtype=float)
    num_inputs_by_layer = num_input_units * numpy.exp(
        -1 * dense_layer_indices / e_folding_param)
    num_inputs_by_layer = numpy.round(num_inputs_by_layer).astype(int)

    num_outputs_by_layer = numpy.concatenate((
        num_inputs_by_layer[1:],
        numpy.array([num_output_units], dtype=int)
    ))

    return num_inputs_by_layer, num_outputs_by_layer


def setup_cnn(num_grid_rows, num_grid_columns):
    """Sets up (but does not train) CNN (convolutional neural net).
    :param num_grid_rows: Number of rows in each predictor image.
    :param num_grid_columns: Number of columns in each predictor image.
    :return: cnn_model_object: Untrained instance of `keras.models.Model`.
    """

    regularizer_object = keras.regularizers.l1_l2(l1=L1_WEIGHT, l2=L2_WEIGHT)

    num_predictors = len(NETCDF_PREDICTOR_NAMES)
    input_layer_object = keras.layers.Input(
        shape=(num_grid_rows, num_grid_columns, num_predictors)
    )

    current_num_filters = None
    current_layer_object = None

    # Add convolutional layers.
    for _ in range(NUM_CONV_LAYER_SETS):
        for _ in range(NUM_CONV_LAYERS_PER_SET):

            if current_num_filters is None:
                current_num_filters = (
                    num_predictors * NUM_PREDICTORS_TO_FIRST_NUM_FILTERS
                )
                this_input_layer_object = input_layer_object

            else:
                current_num_filters *= 2
                this_input_layer_object = current_layer_object

            current_layer_object = keras.layers.Conv2D(
                filters=current_num_filters,
                kernel_size=(NUM_CONV_FILTER_ROWS, NUM_CONV_FILTER_COLUMNS),
                strides=(1, 1), padding='valid', data_format='channels_last',
                dilation_rate=(1, 1), activation=None, use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=regularizer_object
            )(this_input_layer_object)

            current_layer_object = keras.layers.LeakyReLU(
                alpha=SLOPE_FOR_RELU
            )(current_layer_object)

            if CONV_LAYER_DROPOUT_FRACTION is not None:
                current_layer_object = keras.layers.Dropout(
                    rate=CONV_LAYER_DROPOUT_FRACTION
                )(current_layer_object)

            if USE_BATCH_NORMALIZATION:
                current_layer_object = keras.layers.BatchNormalization(
                    axis=-1, center=True, scale=True
                )(current_layer_object)

        current_layer_object = keras.layers.MaxPooling2D(
            pool_size=(NUM_POOLING_ROWS, NUM_POOLING_COLUMNS),
            strides=(NUM_POOLING_ROWS, NUM_POOLING_COLUMNS),
            padding='valid', data_format='channels_last'
        )(current_layer_object)

    these_dimensions = numpy.array(
        current_layer_object.get_shape().as_list()[1:], dtype=int)
    num_features = numpy.prod(these_dimensions)

    current_layer_object = keras.layers.Flatten()(current_layer_object)

    # Add intermediate dense layers.
    _, num_outputs_by_dense_layer = _get_dense_layer_dimensions(
        num_input_units=num_features, num_classes=2,
        num_dense_layers=NUM_DENSE_LAYERS)

    for k in range(NUM_DENSE_LAYERS - 1):
        current_layer_object = keras.layers.Dense(
            num_outputs_by_dense_layer[k], activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=regularizer_object
        )(current_layer_object)

        current_layer_object = keras.layers.LeakyReLU(
            alpha=SLOPE_FOR_RELU
        )(current_layer_object)

        if DENSE_LAYER_DROPOUT_FRACTION is not None:
            current_layer_object = keras.layers.Dropout(
                rate=DENSE_LAYER_DROPOUT_FRACTION
            )(current_layer_object)

        if USE_BATCH_NORMALIZATION:
            current_layer_object = keras.layers.BatchNormalization(
                axis=-1, center=True, scale=True
            )(current_layer_object)

    # Add output layer (also dense).
    current_layer_object = keras.layers.Dense(
        1, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularizer_object
    )(current_layer_object)

    current_layer_object = keras.layers.Activation(
        'sigmoid'
    )(current_layer_object)

    if DENSE_LAYER_DROPOUT_FRACTION is not None and NUM_DENSE_LAYERS == 1:
        current_layer_object = keras.layers.Dropout(
            rate=DENSE_LAYER_DROPOUT_FRACTION
        )(current_layer_object)

    # Put the whole thing together and compile.
    cnn_model_object = keras.models.Model(
        inputs=input_layer_object, outputs=current_layer_object)

    cnn_model_object.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=LIST_OF_METRIC_FUNCTIONS)

    cnn_model_object.summary()
    return cnn_model_object


def train_cnn(
        cnn_model_object, training_file_names, normalization_dict,
        targ_norm_dict, num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, output_model_file_name, range_high, range_low,
        validation_file_names=None, num_validation_batches_per_epoch=None):
    """Trains CNN (convolutional neural net).
    :param cnn_model_object: Untrained instance of `keras.models.Model` (may be
        created by `setup_cnn`).
    :param training_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).
    :param normalization_dict: See doc for `deep_learning_generator`.
    :param num_examples_per_batch: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param validation_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).  If `validation_file_names is None`,
        will omit on-the-fly validation.
    :param num_validation_batches_per_epoch:
        [used only if `validation_file_names is not None`]
        Number of validation batches furnished to model in each epoch.
    :return: cnn_metadata_dict: Dictionary with the following keys.
    cnn_metadata_dict['training_file_names']: See input doc.
    cnn_metadata_dict['normalization_dict']: Same.
    cnn_metadata_dict['binarization_threshold']: Same.
    cnn_metadata_dict['num_examples_per_batch']: Same.
    cnn_metadata_dict['num_training_batches_per_epoch']: Same.
    cnn_metadata_dict['validation_file_names']: Same.
    cnn_metadata_dict['num_validation_batches_per_epoch']: Same.
    """

    _create_directory(file_name=output_model_file_name)

    if validation_file_names is None:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=False, mode='min',
            period=1)
    else:
        checkpoint_object = keras.callbacks.ModelCheckpoint(
            filepath=output_model_file_name, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min',
            period=1)

    list_of_callback_objects = [checkpoint_object]

    cnn_metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        TARGET_DICT_KEY: targ_norm_dict

    }

    training_generator = deep_learning_generator(
        netcdf_file_names=training_file_names,
        num_examples_per_batch=num_examples_per_batch,
        range_high=range_high,
        range_low=range_low,
        normalization_dict_target=targ_norm_dict,
        normalization_dict=normalization_dict
        )

    if validation_file_names is None:
        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return cnn_metadata_dict

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)

    reduce_on_plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_REDUCE_LR,factor = FACTOR_REDUCE_LR,
        patience=NUM_EPOCHS_FOR_REDUCE_LR, verbose=1, mode='min',min_lr=MIN_LR_REDUCE_TO)


    #, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    list_of_callback_objects.append(reduce_on_plateau_object)

    validation_generator = deep_learning_generator(
        netcdf_file_names=validation_file_names,
        num_examples_per_batch=num_examples_per_batch,
        range_high=range_high,
        range_low=range_low,
        normalization_dict_target=targ_norm_dict,
        normalization_dict=normalization_dict
        )

    cnn_model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=list_of_callback_objects, workers=0,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)

    return cnn_metadata_dict

def read_keras_model(hdf5_file_name):
    """Reads Keras model from HDF5 file.
    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT)



def apply_cnn(cnn_model_object, predictor_matrix, verbose=True,
              output_layer_name=None):
    """Applies trained CNN (convolutional neural net) to new data.
    E = number of examples (
    M = number of rows in eachgrid
    N = number of columns in each grid
    C = number of channels (predictor variables)

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param verbose: Boolean flag.  If True, progress messages will be printed.
    :param output_layer_name: Name of output layer.  If
        `output_layer_name is None`, this method will use the actual output
        layer, so will return predictions.  If `output_layer_name is not None`,
        will return "features" (outputs from the given layer).
    If `output_layer_name is None`...
    :return: forecast_probabilities: length-E numpy array with forecast
        probabilities of positive class (label = 1).
    If `output_layer_name is not None`...
    :return: feature_matrix: numpy array of features (outputs from the given
        layer).  There is no guarantee on the shape of this array, except that
        the first axis has length E.
    """

    num_examples = predictor_matrix.shape[0]
    num_examples_per_batch = 40

    if output_layer_name is None:
        model_object_to_use = cnn_model_object
    else:
        model_object_to_use = keras.models.Model(
            inputs=cnn_model_object.input,
            outputs=cnn_model_object.get_layer(name=output_layer_name).output
        )

    output_array = None

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                this_first_index, this_last_index, num_examples
            ))

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        this_output_array = model_object_to_use.predict(
            predictor_matrix[these_indices, ...],
            batch_size=num_examples_per_batch)

        if output_layer_name is None:
            this_output_array = this_output_array

        if output_array is None:
            output_array = this_output_array + 0.
        else:
            output_array = numpy.concatenate(
                (output_array, this_output_array), axis=0
            )

    return output_array


def _init_figure_panels(num_rows, num_columns, horizontal_space_fraction=0.1,
                        vertical_space_fraction=0.1, keep_aspect_ratio=True):
    """Initializes paneled figure.
    :param num_rows: Number of panel rows.
    :param num_columns: Number of panel columns.
    :param horizontal_space_fraction: Horizontal space between panels (as
        fraction of panel size).
    :param vertical_space_fraction: Vertical space between panels (as fraction
        of panel size).
    :param keep_aspect_ratio: Boolean flag.  If True, aspect ratio of each panel
        will be forced to reflect aspect ratio of the data.  For example, if the
        grid plotted in each panel is 32 x 32, each panel must be a square.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        the handle (instance of `matplotlib.axes._subplots.AxesSubplot`) for the
    """

    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_rows, num_columns, sharex=False, sharey=False,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_rows == num_columns == 1:
        axes_objects_2d_list = [[axes_objects_2d_list]]
    elif num_columns == 1:
        axes_objects_2d_list = [[a] for a in axes_objects_2d_list]
    elif num_rows == 1:
        axes_objects_2d_list = [axes_objects_2d_list]

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=vertical_space_fraction, wspace=horizontal_space_fraction)

    if not keep_aspect_ratio:
        return figure_object, axes_objects_2d_list

    for i in range(num_rows):
        for j in range(num_columns):
            axes_objects_2d_list[i][j].set(
                adjustable='box-forced', aspect='equal')

    return figure_object, axes_objects_2d_list