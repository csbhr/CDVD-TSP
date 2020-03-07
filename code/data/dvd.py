import os
from data import videodata


class DVD(videodata.VIDEODATA):
    def __init__(self, args, name='DVD', train=True):
        super(DVD, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'gt')
        self.dir_input = os.path.join(self.apath, 'blur')
        print("DataSet gt path:", self.dir_gt)
        print("DataSet blur path:", self.dir_input)
