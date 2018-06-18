import pickle


def pickle_load(file, **kwargs):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            obj = pickle.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = pickle.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
    return obj


def pickle_dump(obj, file=None, **kwargs):
    kwargs.setdefault('protocol', 2)
    if file is None:
        return pickle.dumps(obj, **kwargs)
    elif isinstance(file, str):
        with open(file, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif hasattr(file, 'write'):
        pickle.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
