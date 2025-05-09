import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
