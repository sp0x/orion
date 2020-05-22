
if __name__ == "__main__":
    import sys
    import os
    import argparse
    import time
    from server import Server
    from ml import classifiers, neural_networks
    from utils import load_json, abs_path, par_path
    from constants import SKLEARN_PARAMS_PATH, SKLEARN_CLASS_PATH, KERAS_CLASS_PATH
    from prediction_server import  PredictionServer
    sys.path.append(os.getcwd())
    import multiprocessing

    cmd_parser = argparse.ArgumentParser(description='Orion ml server')
    cmd_parser.add_argument('--mode', help='Makes the server run in `pred` or `train` only mode')
    args = cmd_parser.parse_args()

    classifiers.model_table = load_json(abs_path(SKLEARN_CLASS_PATH))
    classifiers.param_table = load_json(abs_path(SKLEARN_PARAMS_PATH))
    neural_networks.layer_list = load_json(abs_path(KERAS_CLASS_PATH))
    
    server_mode = 'full'
    if args.mode == 'pred':
        print("Running in prediction only mode.")
        server_mode = 'prediction'
    elif args.mode == 'train':
        print("Running in training only mode.")
        server_mode = 'train'

    #multiprocessing.set_start_method('forkserver')
    if server_mode in ['full', 'train']:
        ORION_SERVER = Server()
        ORION_SERVER.start()    
    if server_mode in ['full', 'prediction']:
        prediction_server = PredictionServer()
        prediction_server.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting server down")
    except Exception:
        pass
    if server_mode in ['full', 'train']:
        ORION_SERVER.shutdown()
