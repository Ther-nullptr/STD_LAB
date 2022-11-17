import yaml

class Config(object):
    def __init__(self, config):
        with open(config, 'r') as stream:
            docs = yaml.load_all(stream, Loader=yaml.FullLoader)
            self.string = ''
            for doc in docs:
                for k, v in doc.items():
                    cmd = "self." + k + "=" + repr(v)
                    self.string += (k + "=" + repr(v) + '_')
                    print(cmd)
                    exec(cmd)

    def __str__(self):
        return self.string


if __name__ == '__main__':
    config = Config('/mnt/c/Users/86181/Desktop/STD-project/configs/train_config.yaml')
    print(str(config))
