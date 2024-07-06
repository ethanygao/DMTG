import copy

models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(model_spec, args=None, load_sd=False):
    model_args = model_spec.get("args")
    if model_args is not None and args is not None:
        model_args = copy.deepcopy(model_args)
        model_args.update(args)
    elif model_args is None:
        model_args = args
    else:
        pass

    if model_args is not None:
        model = models[model_spec["name"]](**model_args)
    else:
        model = models[model_spec["name"]]()
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
