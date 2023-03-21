import torch


class Scheduler(object):
    """Responsible for scheduling pruning and masking parameters.

    """

    def __init__(self, model, device=torch.device("cuda")):
        self.model = model
        self.device = device
        self.policies = {}
        self.sched_metadata = {}

    def add_policy(self, policy, epochs=None, starting_epoch=None, ending_epoch=None, frequency=1):
        """Add a new policy to the schedule.

        Args:
            epochs (list): A list, or range, of epochs in which to apply the policy.
            starting_epoch (integer): An integer number specifying at which epoch to start.
            ending_epoch (integer): An integer number specifying at which epoch to end.
            frequency (integer): An integer number specifying how often to invoke the policy.

            You may only provide a list of `epochs` or a range of epochs using `starting_epoch`
            and `ending_epoch` (i.e. these are mutually-exclusive)
        """
        assert (epochs is None and None not in (starting_epoch, ending_epoch, frequency)) or \
               (epochs is not None and all(c is None for c in (starting_epoch, ending_epoch)))

        if epochs is None:
            assert 0 <= starting_epoch < ending_epoch
            assert 0 < frequency <= (ending_epoch - starting_epoch)
            epochs = list(range(starting_epoch, ending_epoch, frequency))
        else:
            starting_epoch = epochs[0]
            ending_epoch = epochs[-1] + 1
            frequency = None

        for epoch in epochs:
            if epoch not in self.policies:
                self.policies[epoch] = [policy]
            else:
                self.policies[epoch].append(policy)
            assert len(self.policies[epoch]) > 0

        self.sched_metadata[policy] = {'starting_epoch': starting_epoch,
                                       'ending_epoch': ending_epoch,
                                       'frequency': frequency}

    def on_epoch_begin(self, epoch, optimizer=None, **kwargs):
        pass

    def on_minibatch_begin(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        pass

    def on_epoch_end(self, epoch, optimizer=None, **kwargs):
        for policy in self.policies.get(epoch, list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            meta['optimizer'] = optimizer
            policy.on_epoch_end(self.model, meta, **kwargs)
