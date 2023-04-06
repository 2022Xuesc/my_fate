import logging
import logging.config
import time


def func1():
    logging.warning('Watch out!')
    # 由于默认级别是warning，因此info不会出现
    logging.info('I told you so')


def log_to_file1():
    logging.basicConfig(filename='example1.log', filemode='w', level=logging.DEBUG)
    logging.debug('this is debug')


def log_to_file2():
    logging.basicConfig(filename='example2.log', filemode='w', level=logging.DEBUG)
    logging.debug('this is debug')


def log_format():
    epoch = 4
    num = 5
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f'epoch为{epoch},num为{num}')
    logging.warning('a warning')


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf', defaults={'fate_filename': 'fate.log', 'quant_filename': 'quant.log'})
    fate_logger = logging.getLogger('fate_logger')
    import torch

    t = torch.tensor([1, 2, 3])
    fate_logger.debug(t)

    quant_logger = logging.getLogger('quant_logger')
    quant_logger.debug('hello quant1')
