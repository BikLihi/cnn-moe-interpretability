import importlib

def find_model_by_name(model_name):
    file_name = 'models.' + model_name
    lib = importlib.import_module(file_name)
    
    for name, cls in lib.__dict__.items():
        if name.lower() == model_name.replace('_', '').lower():
            return cls
    
    raise RuntimeError('No model found by name ' + model_name)


def build_model(config):
    model = find_model_by_name(config['model_name'])
    model_instance = model(config)
    return model_instance