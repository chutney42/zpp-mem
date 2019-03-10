from collections import OrderedDict
import json

class NoneDict(OrderedDict):
    def __missing__(self, key):
        return None

def parse(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as options_file:
        for line in options_file:
            line = line.split('//')[0] + '\n'
            json_str += line
    options = json.loads(json_str, object_pairs_hook=NoneDict)

    return options
