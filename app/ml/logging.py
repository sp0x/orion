import datetime
import pandas as pd
import numpy as np

import envir


class ExperimentLogger(object):
    def __init__(self, client):
        from helpers import ClientFs
        self.client = client
        self.client_fs = ClientFs(client)
        self.current_target = None
        self.current_file = None

    def get_log_file(self, target):
        experiment_log = "exp_{0}_{1}.log".format(target, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_file = self.client_fs.get(["logs", experiment_log])
        return log_file

    def set_current_target(self, target):
        self.current_target = target
        self.current_file = self.get_log_file(self.current_target)

    def log_report(self, data, target, filename=None):
        log_file = self.get_log_file(target) if filename is None else filename

        #print("Logging to file: " + log_file)
        if 'best_performance' not in data:
            data['best_performance'] = 'na'
        if 'best' not in data:
            data['best'] = 'na'
        if 'accuracy' not in data:
            data['accuracy'] = 'na'
        if 'results' not in data:
            data['results'] = []
        if 'statistics' not in data:
            data['statistics'] = ''
        if 'columns' not in data:
            data['columns'] = {'data': [], 'target': []}
        if 'feature_importance' not in data:
            data['feature_importance'] = ''
        if 'scaling' not in data:
            data['scaling'] = {'data': 'none', 'target': 'none'}
        with open(log_file, 'w') as f:
            report = str(data['model']) + '\n'
            report += "=========Best configuration==============\n"
            report += str(data['best'] if 'best' in data else 0) + '\n'
            report += "\n==========Best model performance on test cases===========\n"
            if isinstance(data['best_performance'] if 'best_performance' in data else None, list):
                report += "\n".join(data['best_performance'])
            else:
                report += data['best_performance']
            report += "\nAccuracy: " + str(data['accuracy']) + "\n"
            report += "\n Feature Importance: +" + str(data['feature_importance'])
            report += "\n==========Other configurations============\n"
            report += pd.DataFrame(data['results']).to_string()
            report += "\n====================Stats===================\n"
            report += data['statistics']
            report += "\n====================Scaling===================\n"
            report += "Data: {0}    Target: {1}\n".format(data['scaling']['data'], data['scaling']['target'])
            report += "\n====================Columns===================\n"
            report += "Data: " + str(data['columns']['data']) + "\n"
            report += "Target: " + str(data['columns']['target']) + "\n"
            report += '\n\n\n'
            report = report.replace('\n', '\r\n')
            #print("Wrote log file..")
            f.write(report)
        return log_file
