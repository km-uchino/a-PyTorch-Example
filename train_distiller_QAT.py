import math
import time
import os
import traceback
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# tensorboardX
from tensorboardX import SummaryWriter
writer = SummaryWriter()

# distiller
import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
from   distiller.data_loggers import *
import distiller.quantization as quantization

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
#checkpoint = None  # path to model checkpoint, None if none
checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
batch_size = 8  # batch size
start_epoch = 0  # start at this epoch
epochs = 190  # number of epochs to run without early-stopping
#epochs = 20  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
best_loss = 100.  # assume a high loss at first
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training or validation status every __ batches
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

# Logger handle
msglogger = None

# classifier_compression ではargsだったもの
args_output_dir = './logs/'
args_name = 'test'
args_compress = 'quant_aware_train_linear_quant.yaml'
args_arch = 'SSD'

def main():
    """
    Training and validation.
    """
    script_dir = os.path.dirname(__file__)
    print('script_dir : {:}'.format(script_dir))
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint
    global msglogger

    ## Parse arguments
    #args = parser.get_parser().parse_args()
    #if args.epochs is None:
    #    args.epochs = 90

    if not os.path.exists(args_output_dir):
        os.makedirs(args_output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args_name, args_output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    #apputils.log_execution_env_state(args.compress, msglogger.logdir, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)
    
    ## Initialize model or load checkpoint
    #if checkpoint is None:
    #    model = SSD300(n_classes=n_classes)
    #    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    #    biases = list()
    #    not_biases = list()
    #    for param_name, param in model.named_parameters():
    #        if param.requires_grad:
    #            if param_name.endswith('.bias'):
    #                biases.append(param)
    #            else:
    #                not_biases.append(param)
    #    #optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
    #    #                            lr=lr, momentum=momentum, weight_decay=weight_decay)
    #    optimizer = torch.optim.SGD(model.parameters(),
    #                                lr=lr, momentum=momentum, weight_decay=weight_decay)
    #
    #else:
    #    checkpoint = torch.load(checkpoint)
    #    start_epoch = checkpoint['epoch'] + 1
    #    epochs_since_improvement = checkpoint['epochs_since_improvement']
    #    best_loss = checkpoint['best_loss']
    #    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    #    model = checkpoint['model']
    #    optimizer = checkpoint['optimizer']
        
    # load checkpoint
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_loss = checkpoint['best_loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    model = checkpoint['model']
    #optimizer = checkpoint['optimizer']
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Move to default device
    model = model.to(device)

    compression_scheduler = None
    # The main use-case for this sample application is CNN compression. 
    # Compression requires a compression schedule configuration file in YAML.
    print('model:{:}'.format(model))
    print('optimizer:{:}'.format(optimizer))
    print('args_compress:{:}'.format(args_compress))
    print('compression_scheduler:{:}'.format(compression_scheduler))
    compression_scheduler = distiller.file_config(model, optimizer, args_compress, compression_scheduler,
                                                  #(start_epoch-1) if args.resumed_checkpoint_path else None)
                                                  None)
    print(compression_scheduler)
    # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
    model.to(device)
    
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    #tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # Define loss function (criterion)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    val_dataset = PascalVOCDataset(data_folder,
                                   split='test',
                                   keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)
    # Epochs
    for epoch in range(start_epoch, epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)

        # In practice, I just decayed the learning rate when loss stopped improving for long periods,
        # and I would resume from the last best checkpoint with the new learning rate,
        # since there's no point in resuming at the most recent and significantly worse checkpoint.
        # So, when you're ready to decay the learning rate, just set checkpoint = 'BEST_checkpoint_ssd300.pth.tar' above
        # and have adjust_learning_rate(optimizer, 0.1) BEFORE this 'for' loop

        ### distiller ###
        compression_scheduler.on_epoch_begin(epoch)  

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              compression_scheduler=compression_scheduler)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)
        writer.add_scalar('data/val_loss', val_loss, epoch)

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        #save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)
        #checkpoint_extras = {'current_top1': top1,
        #                     'best_top1': perf_scores_history[0].top1,
        #                     'best_epoch': perf_scores_history[0].epoch}
        apputils.save_checkpoint(epoch, args_arch, model, optimizer=optimizer, scheduler=compression_scheduler,
                                 #extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)
                                 is_best=is_best, name=args_name, dir=msglogger.logdir)

        ### distiller ###
        compression_scheduler.on_epoch_end(epoch, optimizer)
        
    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()        

#def train(train_loader, model, criterion, optimizer, epoch):
def train(train_loader, model, criterion, optimizer, epoch, compression_scheduler):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        ### distiller ###
        #compression_scheduler.on_minibatch_begin(epoch)
        compression_scheduler.on_minibatch_begin(epoch, i, steps_per_epoch, optimizer)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        ### distiller ###
        #compression_scheduler.before_backward_pass(epoch)
        agg_loss = compression_scheduler.before_backward_pass(epoch, i, steps_per_epoch, loss,
                                                              optimizer=optimizer, return_loss_components=True)
        loss = agg_loss.overall_loss
        
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        compression_scheduler.before_parameter_optimization(epoch, i, steps_per_epoch, optimizer)
        # Update model
        optimizer.step()

        ### distiller ###
        #compression_scheduler.on_minibatch_end(epoch)
        compression_scheduler.on_minibatch_end(epoch, i, steps_per_epoch, optimizer)
        
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
        # test
        if i == print_freq:
            break
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def validate(val_loader, model, criterion):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
            # test
            if i == print_freq:
                break

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


def check_pytorch_version():
    from pkg_resources import parse_version
    print('torch.__version__:{:}'.format(torch.__version__))
    if parse_version(torch.__version__) < parse_version('1.0.1'):
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 1.0.1 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 1.0.1 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)

#if __name__ == '__main__':
#    main()

if __name__ == '__main__':
    try:
        check_pytorch_version()
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
