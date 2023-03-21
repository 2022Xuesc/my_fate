import federatedml.nn.backend.multi_label as multi_label
# 注意这里的导入语句不可省略
from federatedml.nn.backend.multi_label.utils import filter_kwargs
from torch.optim.lr_scheduler import *


# noinspection PyShadowingNames
def config_scheduler(model, optimizer, sched_dict, scheduler=None):
    if not scheduler:
        scheduler = multi_label.Scheduler(model)

    lr_policies = []
    policy = None
    # 这里的policy_def是key
    policies = sched_dict['policies']
    for policy_type in policies:
        policy_def = policies[policy_type]
        if 'lr_scheduler' == policy_type:
            lr_policies.append(policy_def)
            continue
    lr_schedulers = __factory('lr_schedulers', model, sched_dict, optimizer=optimizer, last_epoch=-1)
    for policy_def in lr_policies:
        instance_name, args = __policy_params(policy_def)
        assert instance_name in lr_schedulers, "LR-scheduler {} was not defined in the list of lr-schedulers".format(
            instance_name)
        lr_scheduler = lr_schedulers[instance_name]
        policy = multi_label.LRPolicy(lr_scheduler)
        add_policy_to_scheduler(policy, policy_def, scheduler)
    return scheduler


def __factory(container_type, model, sched_dict, **extra_args):
    container = {}
    if container_type in sched_dict:
        for name, user_args in sched_dict[container_type].items():
            instance = build_component(model, name, user_args, **extra_args)
            container[name] = instance

    return container


def build_component(model, name, user_args, **extra_args):
    class_name = user_args.pop('class')
    class_ = globals()[class_name]
    # First we check that the user defined dict itself does not contain invalid args
    valid_args, invalid_args = filter_kwargs(user_args, class_.__init__)
    if invalid_args:
        raise ValueError(
            '{0} does not accept the following arguments: {1}'.format(class_name, list(invalid_args.keys())))

    # Now we add some "hard-coded" args, which some classes may accept and some may not
    # So then we filter again, this time ignoring any invalid args
    valid_args.update(extra_args)
    valid_args['model'] = model
    valid_args['name'] = name
    final_valid_args, _ = filter_kwargs(valid_args, class_.__init__)
    instance = class_(**final_valid_args)
    return instance


def __policy_params(policy):
    name = policy['instance_name']
    args = policy.get('args', None)
    return name, args


def add_policy_to_scheduler(policy, policy_def, scheduler):
    if 'epochs' in policy_def:
        scheduler.add_policy(policy, epochs=policy_def['epochs'])
    else:
        scheduler.add_policy(policy, starting_epoch=policy_def['starting_epoch'],
                             ending_epoch=policy_def['ending_epoch'],
                             frequency=policy_def['frequency'])
