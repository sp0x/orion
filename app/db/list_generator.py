import inspect
import json
from utils import par_path
from constants import SKLEARN_CLASS_PATH, SKLEARN_PARAMS_PATH, KERAS_CLASS_PATH

# Use to regenerate the classlist.json and paramlist.json files
def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

if __name__ == '__main__':
    from sys import argv
    myargs = getopts(argv)
    namespace = 'sklearn'
    if '-n' in myargs:
        namespace = myargs['-n']
    if namespace == 'keras':
        namespaces = ['keras.layers']
        class_path = KERAS_CLASS_PATH
        param_path = None
    else:
        namespaces = [
            "sklearn.ensemble",
            "sklearn.tree",
            "sklearn.neighbors",
            "sklearn.linear_model",
            "sklearn.naive_bayes",
            "sklearn.semi_supervised",
            "sklearn.svm",
            "sklearn.gaussian_process"
        ]
        class_path = SKLEARN_CLASS_PATH
        param_path = SKLEARN_PARAMS_PATH
    paramlist = {}
    klasslist = {}
    for n in namespaces:
        mod = __import__(n, fromlist=[n.split(".")[1]])
        for name in dir(mod):
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                if param_path:
                    try:
                        ar = inspect.getargspec(obj.__init__)
                        args = ar.args
                        args.remove('self')
                        params = dict(zip(args, ar.defaults))
                        if 'random_state' in params:
                            params.pop('random_state')
                        if 'memory' in params:
                            params.pop('memory')
                        paramlist[name] = params
                    except:
                        pass
                klasslist[name] = n
    with open(par_path(class_path), "w") as f:
        json.dump(klasslist, f, indent=4)
    if param_path:
        with open(par_path(param_path), "w") as fp:
            json.dump(paramlist, fp, indent=4)

