def print_param_sumstats(params, name):
    print('Summary stats for {}'.format(name))
    param = params[name]
    print('Sum: {}'.format(param.sum()))
    print('Mean: {}'.format(param.mean()))
    print('Std: {}'.format(param.std()))
    print('Max: {} Min: {}'.format(param.max(), param.min()))
    print()