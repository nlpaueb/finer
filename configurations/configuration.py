import json
import os
import logging

from shutil import copy2
from time import strftime, gmtime

from configurations import CONFIG_DIR
from data import EXPERIMENTS_RUNS_DIR, VECTORS_DIR

parameters = {}


class ParameterStore(type):
    def __getitem__(cls, key: str):
        global parameters
        return parameters[key]

    def __setitem__(cls, key, value):
        global parameters
        parameters[key] = value

    def __contains__(self, item):
        global parameters
        return item in parameters


class Configuration(object, metaclass=ParameterStore):
    """
    Configuration class that contains all the parameters of the experiment.
    The experiment parameters are loaded from the corresponding configuration file
    (e.g. "finer_transformer.json)

    It instantiates the experiment logger.
    The output of the experiment is saved at the EXPERIMENTS_RUNS_DIR (data/experiments_runs)
    """

    @staticmethod
    def configure(method, mode):
        global parameters

        os.makedirs(name=os.path.join(EXPERIMENTS_RUNS_DIR), exist_ok=True)
        os.makedirs(name=os.path.join(VECTORS_DIR), exist_ok=True)

        if method in ['transformer', 'bilstm']:
            config_filepath = os.path.join(CONFIG_DIR, f'{method}.json')
            if os.path.exists(config_filepath):
                with open(config_filepath) as config_file:
                    parameters = json.load(config_file)
            else:
                raise Exception(f'Configuration file "{method}.json" does not exist')
        else:
            raise NotImplementedError(f'"FINER-139" experiment does not support "{method}" method')

        parameters['task'] = {}
        parameters['task']['model'] = method
        parameters['task']['mode'] = mode
        parameters['configuration_filename'] = f'{method}.json'

        # Setup Logging
        timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        log_name = f"FINER139_{timestamp}"

        # Set experiment_path and create necessary directories
        if mode == 'train':
            experiment_path = os.path.join(EXPERIMENTS_RUNS_DIR, log_name)
            os.makedirs(name=os.path.join(experiment_path, 'model'), exist_ok=True)
            copy2(src=os.path.join(CONFIG_DIR, f'{method}.json'),
                  dst=os.path.join(experiment_path, f'{method}.json'))

        elif mode == 'evaluate':
            pretrained_model = parameters['evaluation']['pretrained_model']
            if pretrained_model is None or pretrained_model.strip() == '':
                raise Exception(f'No pretrained_model provided in configuration')
            if os.path.exists(os.path.join(EXPERIMENTS_RUNS_DIR, pretrained_model, 'model')):
                experiment_path = os.path.join(EXPERIMENTS_RUNS_DIR, pretrained_model)
                pretrained_model_path = os.path.join(experiment_path, 'model')
            else:
                raise Exception(f'Model "{pretrained_model}" does not exist')

            configuration_path = os.path.join(experiment_path, f'{method}.json')
            if os.path.exists(configuration_path):
                with open(configuration_path) as fin:
                    original_parameters = json.load(fin)
                    for key in ['train_parameters', 'general_parameters', 'hyper_parameters']:
                        parameters[key] = original_parameters[key]
                    parameters['task']['mode'] = mode
            else:
                raise Exception(f'Configuration "{configuration_path}" does not exist')

            parameters['evaluation']['pretrained_model_path'] = pretrained_model_path
            log_name = f"{log_name}_EVALUATE_{'_'.join(parameters['evaluation']['pretrained_model'].split(os.sep))}"

        else:
            raise Exception(f'Mode "{mode}" is not supported')

        parameters['task']['log_name'] = log_name
        parameters['experiment_path'] = experiment_path

        # If in debug mode set workers and max_queue_size to minimum and multiprocessing to False
        if parameters['general_parameters']['debug']:
            parameters['general_parameters']['workers'] = 1
            parameters['general_parameters']['max_queue_size'] = 1
            parameters['general_parameters']['use_multiprocessing'] = False
            parameters['general_parameters']['run_eagerly'] = True

        # Clean loggers
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)

        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(experiment_path, f'{log_name}.txt'),
                            filemode='a')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

    @classmethod
    def __getitem__(cls, item: str):
        global parameters
        return parameters[item]
