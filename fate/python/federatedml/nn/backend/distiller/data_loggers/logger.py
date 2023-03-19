import torch

from .tbbackend import TBBackend
from federatedml.nn.backend.distiller.utils import density, sparsity, size_to_str, to_np, \
    norm_filters, sparsity_2D, has_children


class DataLogger(object):
    """This is an abstract interface for data loggers

    Data loggers log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """

    def __init__(self):
        pass

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        pass

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        pass

    def log_weights_sparsity(self, model, epoch):
        pass

    def log_weights_distribution(self, named_params, steps_completed):
        pass

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        pass


NullLogger = DataLogger


class TensorBoardLogger(DataLogger):
    def __init__(self, logdir):
        super(TensorBoardLogger, self).__init__()
        # Set the tensorboard logger
        self.tblogger = TBBackend(logdir)
        print('\n--------------------------------------------------------')
        print('Logging to TensorBoard - remember to execute the server:')
        print('> tensorboard --logdir=\'./logs\'\n')

        # Hard-code these preferences for now
        self.log_gradients = False  # True
        self.logged_params = ['weight']  # ['weight', 'bias']

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        def total_steps(total, epoch, completed):
            return total * epoch + completed

        prefix = stats_dict[0]
        stats_dict = stats_dict[1]

        for tag, value in stats_dict.items():
            self.tblogger.scalar_summary(prefix + tag, value, total_steps(total, epoch, completed))
        self.tblogger.sync_to_file()

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        group = stat_name + '/activations/' + phase + "/"
        for tag, value in activation_stats.items():
            self.tblogger.scalar_summary(group + tag, value, epoch)
        self.tblogger.sync_to_file()

    def log_weights_sparsity(self, model, epoch):
        params_size = 0
        sparse_params_size = 0

        for name, param in model.state_dict().items():
            if param.dim() in [2, 4]:
                _density = density(param)
                params_size += torch.numel(param)
                sparse_params_size += param.numel() * _density
                self.tblogger.scalar_summary('sparsity/weights/' + name,
                                             sparsity(param) * 100, epoch)
                self.tblogger.scalar_summary('sparsity-2D/weights/' + name,
                                             sparsity_2D(param) * 100, epoch)

        self.tblogger.scalar_summary("sparsity/weights/total", 100 * (1 - sparse_params_size / params_size), epoch)
        self.tblogger.sync_to_file()

    def log_weights_filter_magnitude(self, model, epoch, multi_graphs=False):
        """Log the L1-magnitude of the weights tensors.
        """
        for name, param in model.state_dict().items():
            if param.dim() in [4]:
                self.tblogger.list_summary('magnitude/filters/' + name,
                                           list(to_np(norm_filters(param))), epoch, multi_graphs)
        self.tblogger.sync_to_file()

    def log_weights_distribution(self, named_params, steps_completed):
        if named_params is None:
            return
        for tag, value in named_params:
            tag = tag.replace('.', '/')
            if any(substring in tag for substring in self.logged_params):
                self.tblogger.histogram_summary(tag, to_np(value), steps_completed)
            if self.log_gradients:
                self.tblogger.histogram_summary(tag + '/grad', to_np(value.grad), steps_completed)
        self.tblogger.sync_to_file()

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.

        Notes:
            1. Buffers are logged separately per-layer (i.e. module) within model
            2. All values in a single buffer are logged such that they will be displayed on the same graph in
               TensorBoard
            3. Similarly, if multiple buffers are provided in buffer_names, all are presented on the same graph.
               If this is un-desirable, call the function separately for each buffer
            4. USE WITH CAUTION: While sometimes desirable, displaying multiple distinct values in a single
               graph isn't well supported in TensorBoard. It is achieved using a work-around, which slows
               down TensorBoard loading time considerably as the number of distinct values increases.
               Therefore, while not limited, this function is only meant for use with a very limited number of
               buffers and/or values, e.g. 2-5.

        """
        for module_name, module in model.named_modules():
            if has_children(module):
                continue

            sd = module.state_dict()
            values = []
            for buf_name in buffer_names:
                try:
                    values += sd[buf_name].view(-1).tolist()
                except KeyError:
                    continue

            if values:
                tag = '/'.join([tag_prefix, module_name])
                self.tblogger.list_summary(tag, values, total * epoch + completed, len(values) > 1)
        self.tblogger.sync_to_file()
