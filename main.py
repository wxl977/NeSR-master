from util import ModelConf
from data import FileIO
from NeSR import NeSR

class Data_process(object):
    def __init__(self, config):
        self.config = config
        dataset = config['training.set'].split("/")[-2]
        if dataset == 'amazon-book' or dataset == 'last-fm':
            self.training_data = FileIO.load_data_set_diy(config['training.set'], config['model.type'])
            self.test_data = FileIO.load_data_set_diy(config['test.set'], config['model.type'])
        else:
            self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
            self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])

    def execute(self):
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data)'
        eval(recommender).execute()


conf = ModelConf('NeSR.conf')
data = Data_process(conf)
data.execute()





