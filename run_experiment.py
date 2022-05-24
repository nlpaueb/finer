import click
import os
import logging

from configurations.configuration import Configuration
from finer import FINER

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

cli = click.Group()


@cli.command()
@click.option('--method', default='transformer')
@click.option('--mode', default='train')
def run_experiment(method, mode):
    """
    Main function that instantiates and runs a new experiment

    :param method: Method to run ("bilstm", "transformer", "transformer_bilstm")
    :param mode: Mode to run ("train", "evaluate")
    """

    # Instantiate the Configuration class
    Configuration.configure(method=method, mode=mode)

    experiment = FINER()

    def log_parameters(parameters):
        LOGGER.info(f'\n---------------- {parameters.split("_")[0].capitalize()} Parameters ----------------')
        for param_name, value in Configuration[parameters].items():
            if isinstance(value, dict):
                LOGGER.info(f'{param_name}:')
                for p_name, p_value in value.items():
                    LOGGER.info(f'\t{p_name}: {p_value}')
            else:
                LOGGER.info(f'{param_name}: {value}')

    if mode == 'train':
        LOGGER.info('\n---------------- Train ----------------')
        LOGGER.info(f"Log Name: {Configuration['task']['log_name']}")
        for params in ['train_parameters', 'general_parameters', 'hyper_parameters', 'evaluation']:
            log_parameters(parameters=params)
        LOGGER.info('\n')
        experiment.train()
    elif mode == 'evaluate':
        LOGGER.info('\n---------------- Evaluate Pretrained Model ----------------')
        for params in ['train_parameters', 'general_parameters', 'evaluation']:
            log_parameters(parameters=params)
        LOGGER.info('\n')
        experiment.evaluate_pretrained_model()


if __name__ == '__main__':
    run_experiment()
