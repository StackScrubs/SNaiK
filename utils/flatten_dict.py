def flatten_dict(in_dict):
    res_dict = {}
    if type(in_dict) is not dict:
        return res_dict

    for k, v in in_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v

    return res_dict