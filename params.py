import sys
import time
from ConfigParser import ConfigParser
import StringIO

class Params():
    def __init__(self, config_file_path):
        cf = ConfigParser()
        read_from = cf.read(config_file_path)

        print "Loaded configurations from (in order)", read_from

        self.CONFIG = cf
        cf.set('info','config_file', config_file_path)
        if not cf.has_option('info', 'model_id'):
            cf.set('info','model_id',str(int(time.time()))+"_"+cf.get('info', 'name'))

        # Info
        self.EXPERIMENT = cf.get('info', 'experiment')
        self.NAME = cf.get('info', 'name')
        self.MODEL_ID = cf.get('info', 'model_id')

        # Dataset
        self.TEST_DATA = cf.get('dataset', 'test_data')
        self.MODEL_PATH = cf.get('dataset', 'model_path')
        self.OUTPUT = cf.get('dataset', 'output')

    def to_string(self):
        output = StringIO.StringIO()
        self.CONFIG.write(output)
        val = output.gatvalue()
        output.close()
        return val

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            self.CONFIG.write(f)

params = Params(['./params.ini'])
