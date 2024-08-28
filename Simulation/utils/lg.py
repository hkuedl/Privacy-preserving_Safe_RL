from collections import OrderedDict


def add_prefix(log_dict: OrderedDict, prefix: str, divider=''):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix
