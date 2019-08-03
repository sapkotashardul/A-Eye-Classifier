import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy_eye = AverageMeter()
    accuracy_cog = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)

        inputs = Variable(inputs)
        targets = Variable(targets)
        output1, output2 = model(inputs)
        loss1 = criterion(output1, targets)
        loss2 = criterion(output2, targets)
        loss = loss1 + loss2

        acc_eye = calculate_accuracy(output1, targets)

        acc_cog = calculate_accuracy(output2, targets)

        losses.update(loss.data[0], inputs.size(0))

        accuracy_eye.update(acc_eye, inputs.size(0))
        accuracy_cog.update(acc_cog, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc_eye': accuracy_eye.val,
            'acc_cog': accuracy_cog.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc Eye {acc_eye.val:.3f} ({acc_eye.avg:.3f})\t'
              'Acc Cog {acc_cog.val:.3f} ({acc_cog.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc_cog=accuracy_cog,
                  acc_eye=accuracy_eye))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc_eye': accuracy_eye.avg,
        'acc_cog': accuracy_cog.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
