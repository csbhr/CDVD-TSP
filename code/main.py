import torch

import data
import model
import loss
import option
from trainer.trainer_cdvd_tsp import Trainer_CDVD_TSP
from logger import logger
import datetime
import time
from datetime import datetime, time as datetime_time, timedelta

def time_diff(start, end):
    if isinstance(start, datetime_time): # convert to datetime
        assert isinstance(end, datetime_time)
        start, end = [datetime.combine(datetime.min, t) for t in [start, end]]
    if start <= end: # e.g., 10:33:26-11:15:49
        return end - start
    else: # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1) # +day
        assert end > start
        return end - start

if __name__ == '__main__':
    args = option.args
    torch.manual_seed(args.seed)
    chkp = logger.Logger(args)

    if args.task == 'VideoDeblur':
        print("Selected task: {}".format(args.task))
        model = model.Model(args, chkp)
        loss = loss.Loss(args, chkp) if not args.test_only else None
        loader = data.Data(args)
        t = Trainer_CDVD_TSP(args, loader, model, loss, chkp)
        start_time = args.start_time
        print("Start time: {}".format(args.start_time.strftime("%Y-%m-%d %H:%M:%S")))
        while not t.terminate():
            t.train()
            t.test()
    else:
        raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

    chkp.done()
