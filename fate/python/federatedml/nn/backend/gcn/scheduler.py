import torch

from federatedml.nn.backend.multi_label.policy import PolicyLoss, LossComponent


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

    # 在反向传播前计算损失，以便于进行优化?
    def before_backward_pass(self,epoch, loss, return_loss_components=False):
        overall_loss = loss
        loss_components = []
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy_loss = policy.before_backward_pass(self.model, epoch,
                                                          overall_loss)
                if policy_loss is not None:
                    cur_loss_components = self.verify_policy_loss(policy_loss)
                    overall_loss = policy_loss.overall_loss
                    loss_components += cur_loss_components
        if return_loss_components:
            return PolicyLoss(overall_loss, loss_components)
        return overall_loss

    @staticmethod
    def verify_policy_loss(policy_loss):
        if not isinstance(policy_loss, PolicyLoss):
            raise TypeError("A Policy's before_backward_pass must return either None or an instance of " +
                            PolicyLoss.__name__)
        cur_loss_components = policy_loss.loss_components
        if not isinstance(cur_loss_components, list):
            cur_loss_components = [cur_loss_components]
        if not all(isinstance(lc, LossComponent) for lc in cur_loss_components):
            raise TypeError("Expected an instance of " + LossComponent.__name__ +
                            " or a list of such instances")
        return cur_loss_components
