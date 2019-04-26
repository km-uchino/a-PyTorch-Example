from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

from model import SSD300, MultiBoxLoss
# distiller
import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
from   distiller.data_loggers import *
import distiller.quantization as quantization

def my_load_checkpoint(model, chkpt_file, optimizer=None, model_device=None, *, lean_checkpoint=False):
    """Load a pytorch training checkpoint.

    Args:
        model: the pytorch model to which we will load the parameters
        chkpt_file: the checkpoint file
        lean_checkpoint: if set, read into model only 'state_dict' field
        optimizer: [deprecated argument]
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, compression_scheduler, optimizer, start_epoch
    """
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

    #msglogger.info("=> loading checkpoint %s", chkpt_file)
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)

    #msglogger.info('=> Checkpoint contents:\n{}\n'.format(get_contents_table(checkpoint)))
    #if 'extras' in checkpoint:
    #    #msglogger.info("=> Checkpoint['extras'] contents:\n{}\n".format(get_contents_table(checkpoint['extras'])))

    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

    compression_scheduler = None
    normalize_dataparallel_keys = False
    if 'compression_sched' in checkpoint:
        compression_scheduler = distiller.CompressionScheduler(model)
        try:
            compression_scheduler.load_state_dict(checkpoint['compression_sched'], normalize_dataparallel_keys)
        except KeyError as e:
            # A very common source of this KeyError is loading a GPU model on the CPU.
            # We rename all of the DataParallel keys because DataParallel does not execute on the CPU.
            normalize_dataparallel_keys = True
            compression_scheduler.load_state_dict(checkpoint['compression_sched'], normalize_dataparallel_keys)
        #msglogger.info("Loaded compression schedule from checkpoint (epoch {})".format( checkpoint_epoch))
    #else:
    #    #msglogger.info("Warning: compression schedule data does not exist in the checkpoint")

    if 'thinning_recipes' in checkpoint:
        if 'compression_sched' not in checkpoint:
            raise KeyError("Found thinning_recipes key, but missing mandatory key compression_sched")
        #msglogger.info("Loaded a thinning recipe from the checkpoint")
        # Cache the recipes in case we need them later
        model.thinning_recipes = checkpoint['thinning_recipes']
        if normalize_dataparallel_keys:
            model.thinning_recipes = [distiller.get_normalized_recipe(recipe) for recipe in model.thinning_recipes]
        distiller.execute_thinning_recipes_list(model,
                                                compression_scheduler.zeros_mask_dict,
                                                model.thinning_recipes)

    print('type(model):{:}'.format(type(model)))
    if 'quantizer_metadata' in checkpoint:
        #msglogger.info('Loaded quantizer metadata from the checkpoint')
        qmd = checkpoint['quantizer_metadata']
        print('qmd:{:}'.format(qmd))
        print('type(model):{:}'.format(type(model)))
        quantizer = qmd['type'](model, optimizer=optimizer, **qmd['params'])  # <--optimizerを追加
        print('type(model):{:}'.format(type(model)))
        quantizer.prepare_model()

    if normalize_dataparallel_keys:
            checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'])
    if model_device is not None:
        model.to(model_device)

    if lean_checkpoint:
        #msglogger.info("=> loaded 'state_dict' from checkpoint '{}'".format(str(chkpt_file)))
        return (model, None, None, 0)

    def _load_optimizer(cls, src_state_dict, model):
        """Initiate optimizer with model parameters and load src_state_dict"""
        # initiate the dest_optimizer with a dummy learning rate,
        # this is required to support SGD.__init__()
        dest_optimizer = cls(model.parameters(), lr=1)
        dest_optimizer.load_state_dict(src_state_dict)
        return dest_optimizer

    try:
        optimizer = _load_optimizer(checkpoint['optimizer_type'],
            checkpoint['optimizer_state_dict'], model)
    except KeyError:
        if 'optimizer' not in checkpoint:
            raise
        # older checkpoints didn't support this feature
        # they had the 'optimizer' field instead
        optimizer = None

    #if optimizer is not None:
    #    #msglogger.info('Optimizer of type {type} was loaded from checkpoint'.format(     type=type(optimizer)))
    #    #msglogger.info('Optimizer Args: {}'.format(  dict((k,v) for k,v in optimizer.state_dict()['param_groups'][0].items()                 if k != 'params')))
    #else:
    #    #msglogger.warning('Optimizer could not be loaded from checkpoint.')

    #msglogger.info("=> loaded checkpoint '{f}' (epoch {e})".format(f=str(chkpt_file),   e=checkpoint_epoch))
    return (model, compression_scheduler, optimizer, start_epoch)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load model checkpoint
#checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
#checkpoint = torch.load(checkpoint)
#print('checkpoint.keys() : {:}'.format(checkpoint.keys()))
#print('epoch : {:}'.format(checkpoint['epoch']))
#print('best_loss : {:}'.format(checkpoint['best_loss']))
#print('type(model) : {:}'.format(type(checkpoint['model'])))

args_resumed_checkpoint_path = 'BEST_checkpoint_ssd300_QATL6.pth.tar'
checkpoint_distiller = torch.load(args_resumed_checkpoint_path)
print('checkpoint_distiller.keys() : {:}'.format(checkpoint_distiller.keys()))
print('epoch : {:}'.format(checkpoint_distiller['epoch']))
print('arch  : {:}'.format(checkpoint_distiller['arch']))
print('optimizer_type : {:}'.format(checkpoint_distiller['optimizer_type']))
print('quantizer_metadata : {:}'.format(checkpoint_distiller['quantizer_metadata']))
print('type(state_dict) : {:}'.format(type(checkpoint_distiller['state_dict'])))
#print('state_dict.keys() : {:}'.format(checkpoint_distiller['state_dict'].keys()))

n_classes = len(label_map)  # utils.py : number of different types of objects
model = SSD300(n_classes=n_classes)
## a-Pytorch-Tutorial-to-Object-Detectionのcheckpointからmodelを読み込む
#model = checkpoint['model']
# distillerで保存したcheckpointからmodelなどを読み込む
#model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
#    model, args_resumed_checkpoint_path, model_device=device)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.001, momentum=0.9, weight_decay=0.0001)
print('optimizer:{:}'.format(optimizer))
model, _, _, start_epoch = my_load_checkpoint(model, args_resumed_checkpoint_path, optimizer=optimizer,
                                              model_device=device, lean_checkpoint=True)
#model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
#    model, args_resumed_checkpoint_path, optimizer=optimizer, model_device=device, lean_checkpoint=True)
##print(checkpoint)
#start_epoch = checkpoint['epoch'] + 1
##best_loss = checkpoint['best_loss']
#model = checkpoint['model']
#model = model.to(device)
print('type(model):{:}'.format(type(model)))
model.eval()  # eval mode disables dropout
print('\nLoaded checkpoint from epoch %d.\n' % (start_epoch))

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])









def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    #img_path = '/media/ssd/ssd data/VOC2007/JPEGImages/000001.jpg'
    img_path = '/home/uchino/export/datasets/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
