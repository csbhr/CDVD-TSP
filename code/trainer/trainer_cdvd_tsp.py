"""This is used to train."""
import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from trainer.trainer import Trainer
from datetime import datetime
from dateutil.relativedelta import relativedelta

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{:02d}:{:02d}:{:02d}'.format(t_diff.hours, t_diff.minutes, t_diff.seconds)

class Trainer_CDVD_TSP(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_CDVD_TSP, self).__init__(
            args, loader, my_model, my_loss, ckp)
        print("Using Trainer-CDVD-TSP")
        assert args.n_sequence == 5, \
            "Only support args.n_sequence=5; but get args.n_sequence={}".format(
                args.n_sequence)

    #def make_optimizer(self):
    #    kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
    #    return optim.Adam([{"params": self.model.get_model().recons_net.parameters()},
    #                       {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6}],
    #                      **kwargs)

    def train(self):
        print("Now training")
        # Starting with PyTorch 1.1.0 and later optimizer.step() must be called before scheduler.step()
        self.optimizer.step()
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        job_epochs_completed = self.args.epochs_completed
        job_epochs_started = job_epochs_completed + 1
        job_total_epochs = self.args.epochs - (epoch - job_epochs_started)
        train_start_time = datetime.now()
        batch_start_time = datetime.now()

        # get_lr was deprecated starting with PyTorch 1.4.0
        try:
            lr = self.scheduler.get_last_lr()[0]
        except:        
            lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('\tEpoch {:4d} with Lr {:.2e} started at {}'.format(
            epoch,
            decimal.Decimal(lr),
            train_start_time.strftime("%Y-%m-%d %H:%M:%S")))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        self.ckp.write_log('\t\t[L1:     Mean Absolute Error(L1 Loss)         ]')
        self.ckp.write_log('\t\t[HEM:    Hard Example Mining                  ]')
        self.ckp.write_log('\t\t[Total:  L1+HEM                               ]')
        self.ckp.write_log('\t\t[now:    Current Time                         ]')
        self.ckp.write_log('\t\t[CeFin:  Estimated Completion of Current epoch]')
        self.ckp.write_log('\t\t[AeFin:  Estimated Completion of all epochs   ]')

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input.to(self.device)
            gt_list = [gt[:, i, :, :, :] for i in range(self.args.n_sequence)]
            gt = torch.cat([gt_list[1], gt_list[2], gt_list[3],
                           gt_list[2]], dim=1).to(self.device)

            recons_1, recons_2, recons_3, recons_2_iter, mid_loss = self.model(
                input)
            output = torch.cat(
                [recons_1, recons_2, recons_3, recons_2_iter], dim=1)

            self.optimizer.zero_grad()
            loss = self.loss(output, gt)
            # mid_loss is not set. This does not appear to have been implemented.
            #if mid_loss:  # mid loss is the loss during the model
            #    loss = loss + self.args.mid_loss_weight * mid_loss
            #    mid_loss_sum = mid_loss_sum + mid_loss.item()
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())
            

            if (batch + 1) % self.args.print_every == 0:
                current_time = datetime.now()
                elapsed_time = current_time - train_start_time
                frames_completed = (batch + 1) * self.args.batch_size
                frames_total = len(self.loader_train.dataset)
                train_end_time = train_start_time + (frames_total * (elapsed_time / frames_completed))
                epoch_end_time = train_end_time

                if job_epochs_completed > 0:
                    job_total_time = self.args.total_train_time + self.args.total_test_time
                    epoch_end_time += self.args.total_test_time/job_epochs_completed
                    job_end_time = self.args.start_time + (job_total_epochs * (job_total_time/job_epochs_completed))
                    job_end_time += epoch_end_time - train_start_time
                    
                else:
                    job_end_time = self.args.start_time + (train_end_time - train_start_time)*job_total_epochs

                # Etienne66 Cleaned up by removing unnecessary fields mid was never utilized and 
                # the total is already included in the display_loss.
                self.ckp.write_log('[{}/{}]\tLoss : {}[now: {}][CeFin: {}][AeFin: {}]'.format(
                    frames_completed,
                    frames_total,
                    self.loss.display_loss(batch),
                    current_time.strftime("%H:%M:%S"),
                    epoch_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    job_end_time.strftime("%Y-%m-%d %H:%M:%S")
                ))
                batch_start_time = datetime.now()

        self.args.epochs_completed += 1
        self.loss.end_log(len(self.loader_train))
        self.args.total_train_time += (datetime.now() - train_start_time)

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('Now testing')
        self.ckp.write_log('\tTest Epoch {:3d}'.format(epoch))
        test_start_time = datetime.now()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            total_PSNR_iter1 = 0.
            total_num = 0.
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                input_center = input[:, self.args.n_sequence // 2, :, :, :]
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                recons_1, recons_2, recons_3, recons_2_iter, _ = self.model(
                    input)

                PSNR_iter1 = utils.calc_psnr(
                    gt, recons_2, rgb_range=self.args.rgb_range)
                total_PSNR_iter1 += PSNR_iter1
                total_num += 1
                PSNR = utils.calc_psnr(
                    gt, recons_2_iter, rgb_range=self.args.rgb_range)
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    gt, input_center, recons_2, recons_2_iter = \
                        utils.postprocess(gt, input_center, recons_2,
                                          recons_2_iter,
                                          rgb_range=self.args.rgb_range,
                                          ycbcr_flag=False, device=self.device)
                    save_list = [gt, input_center, recons_2, recons_2_iter]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR_iter1: {:.3f} PSNR_iter2: {:.3f} '\
                               '(Best: {:.3f} @epoch {})'.format(self.args.data_test,
                                                                 total_PSNR_iter1 / total_num,
                                                                 self.ckp.psnr_log[-1],
                                                                 best[0], best[1] + 1))
            self.args.total_test_time += datetime.now() - test_start_time

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
