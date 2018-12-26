from statistics import mean
from storm.util import Handles
import numpy as np
import torch
import cv2
import progressbar
import PySimpleGUI as sg
from tensorboardX import SummaryWriter
from storm.vis.hooks import hooks

def print_loss_term(key, value):
    print('%s %f' % (key, value.item()))

def print_loss(args):
    print('Loss %f' % args.loss.item())


class TB:

    def __init__(self):
        pass

    def tb_train_loss_term(self, loss, loss_term, value):
        loss.run.tb.add_scalar('loss/train/' + loss_term, value.item(), loss.run.step)

    def tb_test_loss_term(self, loss, loss_term, value):
        loss.run.tb.add_scalar('loss/test/' + loss_term, value.item(), loss.run.step)

    def tb_train_loss(self, trainer, args):
        args.run.tb.add_scalar('loss/Train Loss', args.loss.item(), args.run.step)

    def tb_test_loss(self, tester, args):
        args.run.tb.add_scalar('loss/Test Loss', args.loss.item(), args.run.step)

    def tb_image(self, caller, args):
        if args.run.step % 200 == 0:
            input_data = args.input_data
            output_data = args.output_data
            if isinstance(input_data, tuple):
                input_image = input_data[0][0, 0:3].data
            else:
                input_image = input_data[0, 0:3].data
            if isinstance(output_data, tuple):
                output_image = output_data[0][0, 0:3].data
            else:
                output_image = output_data[0, 0:3].data

            args.run.tb.add_image('input', input_image, args.run.step)
            args.run.tb.add_image('output', output_image, args.run.step)

    def write_histogram(self, caller, epoch):
        z = np.concatenate(epoch.context['zl'], axis=0).squeeze()
        histograms = np.rollaxis(z, 1)
        for i, histogram in enumerate(histograms):
            epoch.run.tb.add_histogram('latentvar' + str(i), histogram, epoch.run.step)


def register_tb(run, config):
    run.tb = SummaryWriter(config.tb_run_dir(run.run_id))
    tb = TB()
    handles = Handles()
    if run.loss is not None:
        handles += run.trainer.register_after_hook(tb.tb_train_loss)
        handles += run.tester.register_after_hook(tb.tb_test_loss)
        handles += run.loss.register_hook(tb.tb_train_loss_term)
    handles += run.trainer.register_after_hook(tb.tb_image)
    handles += run.tester.register_after_hook(tb.tb_image)
    #run.tb.add_graph(run.model, (run.data_package.dataset[0][0].cpu().unsqueeze(0),))
    return handles


class LatentInstrument:
    def __init__(self):
        self.corr_viewer = UniImageViewer('correlation_matrix', (160, 160))

    def store_latent_vars_in_epoch(self, model, input, output):
        if 'zl' not in model.run.epoch.context:
            model.run.epoch.context['zl'] = []
        model.run.epoch.context['zl'].append(output[0].data.cpu().numpy())

    def write_correlation(self, epoch):
        z = np.concatenate(epoch.context['zl'], axis=0).squeeze()
        corr = np.corrcoef(z, rowvar=False)
        corr = np.absolute(corr)
        epoch.run.tb.add_scalar('z/z_ave_correlation', (corr - np.identity(corr.shape[0])).mean(), epoch.run.step)
        corr = np.expand_dims(corr, axis=0)
        self.corr_viewer.render(corr)
        epoch.run.tb.add_image('corr_matrix', corr, epoch.run.step)

def to_numpyRGB(image, invert_color=False):
    """
    Universal method to detect and convert an image to numpy RGB format
    :params image: the output image
    :params invert_color: perform RGB -> BGR convert
    :return: the output image
    """
    if type(image) == torch.Tensor:
        image = image.cpu().detach().numpy()
    # remove batch dimension
    if len(image.shape) == 4:
        image = image[0]
    smallest_index = None
    if len(image.shape) == 3:
        smallest = min(image.shape[0], image.shape[1], image.shape[2])
        smallest_index = image.shape.index(smallest)
    elif len(image.shape) == 2:
        smallest = 0
    else:
        raise Exception(f'too many dimensions, I got {len(image.shape)} dimensions, give me less dimensions')
    if smallest == 3:
        if smallest_index == 2:
            pass
        elif smallest_index == 0:
            image = np.transpose(image, [1, 2, 0])
        elif smallest_index == 1:
            # unlikely
            raise Exception(f'Is this a color image?')
        if invert_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif smallest == 1:
        image = np.squeeze(image)
    elif smallest == 0:
        # greyscale
        pass
    elif smallest == 4:
        # that funny format with 4 color dims
        pass
    else:
        raise Exception(f'dont know how to display color of dimension {smallest}')
    return image


class UniImageViewer:
    def __init__(self, title='title', screen_resolution=(640, 480), format=None, channels=None, invert_color=True):
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution
        self.format = format
        self.channels = channels
        self.invert_color = invert_color

    def render(self, image):

        image = to_numpyRGB(image, self.invert_color)

        image = cv2.resize(image, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, image)
        cv2.waitKey(1)

    def view_input(self, model, input, output):
        image = input[0] if isinstance(input, tuple) else input
        self.render(image)

    def view_output(self, model, input, output):
        image = output[0] if isinstance(output, tuple) else output
        self.render(image)

    def update(self, image):
        self.render(image)


class ProgressMeter:
    """
    Display progress.
    """
    def __init__(self, description, total):
        widgets = [
            progressbar.Percentage(),
            progressbar.Bar(),
            progressbar.DynamicMessage('loss'),
        ]
        import tqdm
        self.losses = []
        self.description = description
        import sys
        #self.tqdm = tqdm.tqdm(total=total)
        self.bar = progressbar.ProgressBar(max_value=total)
        self.losses = []
        self.current = 1

    def update(self, epoch, loss):
        self.losses.append(loss.item())
        #self.tqdm.set_description(f'{self.description} epoch: {epoch} loss : {mean(self.losses)}')
        #self.tqdm.update(1)
        #self.bar.update(self.current, loss=loss)
        self.bar.update(self.current)
        self.current += 1
        #import time
        #time.sleep(0.1)

    def loss(self):
        return mean(self.losses)

    @staticmethod
    def register_after(args):
        if 'progress_meter' not in args.self.context:
            total = len(args.dataloader)
            args.self.context['progress_meter'] = ProgressMeter('train', total=total)
        args.self.context['progress_meter'].update(args.run.epoch.ix, args.loss)

    @staticmethod
    def end_epoch(args):
        if 'progress_meter' in args.run.trainer.context:
            del args.run.trainer.context['progress_meter']
        if 'progress_meter' in args.run.tester.context:
            del args.run.tester.context['progress_meter']


class GUIProgressMeter:
    """
    Display progress.
    """
    def __init__(self, description):
        self.train_losses = []
        self.description = description
        self.test_losses = []
        self.best_test_loss = None
        self.train_current = 0
        self.test_current = 0
        self.window = None
        layout = [
            [sg.Text('epochs'), sg.ProgressBar(0, orientation='h', size=(20, 20), key='epoch_progressbar')],
            [sg.Text('training'), sg.ProgressBar(0, orientation='h', size=(20, 20), key='train_progressbar')],
            [sg.Text('testing'), sg.ProgressBar(0, orientation='h', size=(20, 20), key='test_progressbar')],
            [sg.Text('train_loss: 0.0000', size=(15, 1), key='train_loss_txt'), sg.Text('test_loss: 0.0000', size=(15, 1), key='test_loss_txt')],
            [sg.Text('best test loss : 0.0000', size=(30, 1), key='best_test_loss')],
            [sg.Cancel()]
        ]

        self.window = sg.Window(f'{self.description}').Layout(layout)
        self.epoch_pb = self.window.FindElement('epoch_progressbar')
        self.train_pb = self.window.FindElement('train_progressbar')
        self.test_pb = self.window.FindElement('test_progressbar')
        self.test_loss_txt = self.window.FindElement('test_loss_txt')
        self.train_loss_txt = self.window.FindElement('train_loss_txt')
        self.best_test_loss_txt = self.window.FindElement('best_test_loss')
        self.register()

    def register(self):
        hooks.train_end.register(self.update_train)
        hooks.test_end.register(self.update_test)
        hooks.epoch_end.register(self.end_epoch)

    def update_train(self, current_batch, batch_total, loss, **kwargs):
        self.train_losses.append(loss.item())
        event, values = self.window.Read(timeout=0)
        self.train_pb.UpdateBar(self.train_current + 1, batch_total)
        self.train_loss_txt.Update(f'train_loss: {mean(self.train_losses):.4f}')
        self.train_current += 1

    def update_test(self, current_batch, batch_total, loss, **kwargs):
        self.test_losses.append(loss.item())
        event, values = self.window.Read(timeout=0)
        self.test_pb.UpdateBar(self.test_current + 1, batch_total)
        self.test_loss_txt.Update(f'test_loss: {mean(self.test_losses):.4f}')
        self.test_current += 1

    def end_epoch(self, epoch, total_epochs, **kwargs):
        test_loss = mean(self.test_losses)
        if self.best_test_loss is None:
            self.best_test_loss = test_loss
        elif test_loss < self.best_test_loss:
            self.best_test_loss = test_loss

        self.train_current = 0
        self.test_current = 0
        self.train_losses = []
        self.test_losses = []
        event, values = self.window.Read(timeout=0)
        self.train_pb.UpdateBar(0, 10)
        self.test_pb.UpdateBar(0, 10)
        self.epoch_pb.UpdateBar(epoch + 1, total_epochs)
        self.best_test_loss_txt.Update(f'best test loss: {self.best_test_loss:.4f} at epoch {epoch}')

