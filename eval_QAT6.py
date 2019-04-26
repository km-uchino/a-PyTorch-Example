from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

from model import SSD300, MultiBoxLoss
# distiller
import distiller

# ref. distiller/distiller/apputils/checkpoint.py load_checkpoint()
def my_load_checkpoint(model, chkpt_file, optimizer=None, model_device=None, *, lean_checkpoint=False):
    """Load a pytorch training checkpoint.

    Args:
        model: the pytorch model to which we will load the parameters
        chkpt_file: the checkpoint file
        optimizer: [deprecated argument]
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model
    """
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)

    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

    compression_scheduler = None
    normalize_dataparallel_keys = False

    if 'quantizer_metadata' in checkpoint:
        qmd = checkpoint['quantizer_metadata']
        quantizer = qmd['type'](model, optimizer=optimizer, **qmd['params'])  # <--optimizerを追加
        quantizer.prepare_model()

    model.load_state_dict(checkpoint['state_dict'])
    if model_device is not None:
        model.to(model_device)

    return model


# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 32
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint = './BEST_checkpoint_ssd300.pth.tar'
args_resumed_checkpoint_path = 'BEST_checkpoint_ssd300_QATL6.pth.tar'

# Load model checkpoint that is to be evaluated
# とりあえず、ベースのSSD300モデルをロード
n_classes = len(label_map)  # utils.py : number of different types of objects
model = SSD300(n_classes=n_classes)
# 使わないけどoptimizerを作成 (my_load_checkpointにoptimizerを渡さないとエラーになるので)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
# checkpointに保存された読み込む
model = my_load_checkpoint(model, args_resumed_checkpoint_path, optimizer=optimizer, model_device=device)
# Switch to eval mode
model.eval()                # eval mode disables dropout

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
