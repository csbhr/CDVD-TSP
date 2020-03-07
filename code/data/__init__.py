from importlib import import_module
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        # load training dataset
        if not self.args.test_only:
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train.upper())(self.args, name=self.data_train, train=True)
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu,
                num_workers=self.args.n_threads
            )
        else:
            self.loader_train = None

        # load testing dataset
        m_test = import_module('data.' + self.data_test.lower())
        testset = getattr(m_test, self.data_test.upper())(self.args, name=self.data_test, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.args.cpu,
            num_workers=self.args.n_threads
        )
