'''
Network 2.2 is designed for hyperparameter sweeps, but is more robust for many purposes.
Used proper train/valid/test split
'''

from time import time
from sys import exit
START_TIME = time()

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchmetrics.classification import AUROC, BinaryAUROC
    import numpy as np
    import matplotlib.pyplot as plt
    from pprint import pprint
    from pathlib import Path
    from datetime import timedelta
    from gc import collect
    import signal
    import argparse as ap
    import subprocess as sp
    from warnings import filterwarnings
except ModuleNotFoundError:
    print("ERROR: Necessary modules not found!")
    exit(1)

assert int(torch.__version__.split(".")[0]) >= 2, "ERROR: Must use pythorch version >= 2.0"

# default global variables
SEED = 403    # for replication
MODEL_PATH_NAME = "model" # where the saved temp files will go
MODEL_NAME = "resnet_beta" # the file name
BATCH_SIZE = 1024
MAX_EPOCHS = 25
LR = 0.001  # Initial learning rate
WD = 0.0001 # Weight decay rate
UNCERTAIN = 0 # Value for uncertainty label
EVAL_FREQ = 1 # Output accuracy frequency in epoches
SAVE_FREQ = 25 # Save model frequency
VALID_PER = 0.2 # percentage that goes to valid set
TAGS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
THRESHOLDS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# cacluate the time since start of program
def time_passed(start_time = START_TIME):
    return str(timedelta(seconds=int(time() - start_time)))

# exit the program when wall-time is about to reach
def early_leave(signum, frame):
    print("WARNING: Wall-time about to reach, exitting early now")
    model_save()
    print("WARNING: Training terminated at time: {}".format(time_passed()))
    exit(0)

# tracks the CUDA memory usage, in GB
def memory_used(prt = True):
    result = sp.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=sp.PIPE, stderr=sp.PIPE
    )
    output = result.stdout.decode('utf-8').strip()
    used, total = map(int, output.split(','))
    used = used / 1024.0
    total = total / 1024.0
    if prt:
        print("Using {:.2f} GB of of CUDA memory, {:.2f}% of total".format(used, used / total * 100))
    else:
        return (used, total)

def setup():
    signal.signal(signal.SIGTERM, early_leave)
    filterwarnings("ignore", category = UserWarning)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    global MODEL_PATH
    MODEL_PATH = Path(MODEL_PATH_NAME)
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    print("Models will be saved to ./{} with name {}".format(MODEL_PATH_NAME, MODEL_NAME))
    print("Using pytorch version {}".format(torch.__version__))
    torch_version = torch.__version__.split(".")
    if int(torch_version[0]) < 2:
        print("ERROR: Must use PyTorch version >= 2.0")
        exit(1)

    global DEVICE
    if torch.cuda.is_available():
        DEVICE = "cuda"
        _, mem = memory_used(prt = False)
        print("Using CUDA version {}, {:.2f} GB of CUDA memory available".format(torch.version.cuda, mem))
        cuda_version = torch.version.cuda.split(".")
        v1 = int(cuda_version[0])
        v2 = int(cuda_version[1])
        if v1 < 11 or (v1 == 11 and v2 < 8):
            print("ERROR: Must use CUDA version >= 11.8")
            exit(1)
        
        if torch.cuda.device_count() > 1:
            print("WARNING: DPP parallelization on several GPUs not implemented yet")
    else:
        DEVICE = "cpu"
        print("Using CPU")
        print("WARNING: Using CPU only could lead to low efficiency")
    print("Finished setup at time: {}\n".format(time_passed()))

# convert data from numpy arrays to torch tensors, could specify a data type
def to_torch(data, type = torch.float32):
    return torch.from_numpy(data).type(type)

# load the training and test sets, from np arrays to torch tensors, in NCHW format.
# Use small option to use small set for testing
def load_data(small = False):
    global train_dataloader
    global valid_dataloader
    global test

    if small:
        train_images = np.load("images_frontal_male_train_small.npy") 
        train_tags = np.load("tags_frontal_male_train_small.npy")
    else:
        train_images = np.load("/scratch/midway2/zongqi/images_frontal_train.npy") 
        train_tags = np.load("/scratch/midway2/zongqi/tags_frontal_train.npy")

    print("Changing uncertain to {}".format(UNCERTAIN))
    n_data = train_images.shape[0]
    train_images = to_torch(train_images).unsqueeze(dim = 1) # create dim for channel
    train_tags[train_tags == -1] = UNCERTAIN
    train_tags = to_torch(train_tags)
    boundary = int(n_data * (1 - VALID_PER))
    print("Sucessfully loaded training and valid data, {} and {} images".format(boundary, n_data - boundary))
    train = TensorDataset(train_images[: boundary], train_tags[: boundary])
    valid = TensorDataset(train_images[boundary + 1:], train_tags[boundary + 1:])

    test_images = np.load("images_frontal_male_test.npy")
    test_tags = np.load("tags_frontal_male_test.npy")
    test_tags[test_tags == -1] = UNCERTAIN
    test_images = to_torch(test_images).unsqueeze(dim = 1)
    test_tags = to_torch(test_tags)
    test = TensorDataset(test_images, test_tags)
    print("Sucessfully loaded testing data, {} images in total".format(test_images.shape[0]))

    train_dataloader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
    valid_dataloader = DataLoader(valid, batch_size = BATCH_SIZE, shuffle = True)
    print("Finished loading data at time: {}\n".format(time_passed()))

class picture(object):
    '''
    A picture input with its tag for training. Image stored as 368*320 matrix, tag stored as an array
    '''

    def __init__(self, image, tag = None):
        self.image = image
        self.tag = tag
        '''
        Tags are: (1 positive 0 negative -1 uncertain nan no label)
        No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
        Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
        Pleural Effusion, Pleural Other, Fracture
        ''' 

    # visualize a single graph.
    def visualize(self):
        image = self.image * 255
        plt.figure(figsize = (4, 3))
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()

# Construct a convolutional layer
def conv_layer_cons(in_channels, out_channels, kernel_size = 5, stride = 1, padding = 0):
    return nn.Sequential(
                nn.Conv2d(in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding),
                nn.ReLU()
                )

# Construct the INSIDE of a residual block that keeps num_channels constant
# structure: BN -> ReLU -> Conv -> BN -> Conv. Res connection need to be ADDED manually
def res_block_cons(num_channels, kernel_size = 3, stride = 1, padding = "same"):
    return nn.Sequential(
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels = num_channels,
                  out_channels = num_channels,
                  kernel_size = kernel_size, 
                  stride = stride,
                  padding = padding),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels = num_channels,
                  out_channels = num_channels,
                  kernel_size = kernel_size, 
                  stride = stride,
                  padding = padding)
    )

# Construct the INSIDE of a bottleneck residual block that keeps num_channels constant
# structure: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv. Res connection need to be ADDED manually
def bottleneck_block_cons(in_channels, mid_channels = None, kernel_size = 3, padding = "same"):
    if not mid_channels:
        mid_channels = in_channels // 4
    
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels = in_channels,
                  out_channels = mid_channels,
                  kernel_size = 1, 
                  stride = 1,
                  padding = 0),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels = mid_channels,
                  out_channels = mid_channels,
                  kernel_size = kernel_size, 
                  stride = 1,
                  padding = padding),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels = mid_channels,
                  out_channels = in_channels,
                  kernel_size = 1, 
                  stride = 1,
                  padding = 0),
    )

class res_cnn(nn.Module):
    '''
    A deep CNN with skip connections.
    '''

    def __init__(self):
        super().__init__()
        # intial subsampling and max pooling, 1*320*370 -> 32*157*182 -> 32*78*91
        self.block_1 = conv_layer_cons(in_channels = 1, out_channels = 32, kernel_size = 7, stride = 2)
        self.block_2 = nn.MaxPool2d(kernel_size = 2)

        # block 3-7 are resblocks, 32*78*91 -> 32*78*91
        self.block_3 = res_block_cons(32)
        self.block_4 = res_block_cons(32)
        self.block_5 = res_block_cons(32)
        self.block_6 = res_block_cons(32)
        self.block_7 = res_block_cons(32)

        # subsampling, 32*78*91 -> 64*37*44
        self.block_8 = conv_layer_cons(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 2)

        # block 9-14 are resblocks and block 11-13 are bottleneck blocks, 64*37*44 -> 64*37*44
        self.block_9 = res_block_cons(64)
        self.block_10 = res_block_cons(64)
        self.block_11 = res_block_cons(64)
        self.block_12 = bottleneck_block_cons(64)
        self.block_13 = bottleneck_block_cons(64)
        self.block_14 = bottleneck_block_cons(64)

        # subsampling, 64*37*44 -> 128*17*20
        self.block_15 = conv_layer_cons(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2)

        # block 16-20 are bottleneck blocks, 128*17*20 -> 128*17*20
        self.block_16 = bottleneck_block_cons(128)
        self.block_17 = bottleneck_block_cons(128)
        self.block_18 = bottleneck_block_cons(128)
        self.block_19 = bottleneck_block_cons(128)
        self.block_20 = bottleneck_block_cons(128)

        # subsampling, 128*17*20 -> 256*7*8
        self.block_21 = conv_layer_cons(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 2)

        # block 22-26 are resblocks, 256*7*8 -> 256*7*8
        self.block_22 = bottleneck_block_cons(256)
        self.block_23 = bottleneck_block_cons(256)
        self.block_24 = bottleneck_block_cons(256)
        self.block_25 = bottleneck_block_cons(256)
        self.block_26 = bottleneck_block_cons(256)

        # layer 27 uses average pooling for each dimension, then several fully connected layers with dropout
        # 256*7*8 -> 256 -> 1000 -> 500 -> 14
        self.block_27 = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(in_features = 256, out_features = 1000),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(in_features = 1000, out_features = 500),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(in_features = 500, out_features = 14)
        )
    
    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)

        out_3 = self.block_3(out_2) + out_2
        out_4 = self.block_4(out_3) + out_3
        out_5 = self.block_5(out_4) + out_4
        out_6 = self.block_6(out_5) + out_5
        out_7 = self.block_7(out_6) + out_6
        
        out_8 = self.block_8(out_7)
        out_9 = self.block_9(out_8) + out_8
        out_10 = self.block_10(out_9) + out_9
        out_11 = self.block_11(out_10) + out_10
        out_12 = self.block_12(out_11) + out_11
        out_13 = self.block_13(out_12) + out_12
        out_14 = self.block_14(out_13) + out_13
        
        out_15 = self.block_15(out_14)
        out_16 = self.block_16(out_15) + out_15
        out_17 = self.block_17(out_16) + out_16
        out_18 = self.block_18(out_17) + out_17
        out_19 = self.block_19(out_18) + out_18
        out_20 = self.block_20(out_19) + out_19

        out_21 = self.block_21(out_20)
        out_22 = self.block_22(out_21) + out_21
        out_23 = self.block_23(out_22) + out_22
        out_24 = self.block_24(out_23) + out_23
        out_25 = self.block_25(out_24) + out_24
        out_26 = self.block_26(out_25) + out_25

        out_27 = self.block_27(out_26)
        return out_27

# construct the model
def model_cons():
    global model
    model = res_cnn()
    model.to(DEVICE)
    model = torch.compile(model)

    global scheduler
    global optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WD)
    print("Using initial learning rate {} and weight decay constant {}".format(LR, WD))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode = "max", factor = 0.5, patience = 5)

    global loss_fn
    loss_fn = nn.BCEWithLogitsLoss(reduction = "none")

    global auroc
    global auroc_list
    auroc = AUROC(num_labels = 14, average = "macro", task = "multilabel").to(DEVICE)
    auroc_list = [BinaryAUROC().to(DEVICE) for _ in range(14)]
    print("Finished model construction at time: {}\n".format(time_passed()))

# save a model
def model_save(epoch = None):
    if epoch is not None:
        MODEL_NAME_SAVE = "{}.restart{}.pth".format(MODEL_NAME, epoch) 
    else:
        MODEL_NAME_SAVE = "{}_save.pth".format(MODEL_NAME) 

    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME_SAVE
    torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)
    print("Successfully saved model to {}".format(MODEL_SAVE_PATH))

# load a model
def model_load(name = None, epoch = None):
    if name is not None:
        MODEL_LOAD_PATH = MODEL_PATH / name
    elif epoch is not None:
        MODEL_NAME_LOAD = "{}.restart{}.pth".format(MODEL_NAME, epoch) 
        MODEL_LOAD_PATH = MODEL_PATH / MODEL_NAME_LOAD
    else:
        print("ERROR: must specify file name or restart epoch number")
    
    try:
        model.load_state_dict(torch.load(f = MODEL_LOAD_PATH))
    except FileNotFoundError:
        print("ERROR: must specify file name or restart epoch number")
    except RuntimeError:
        try: 
            model.load_state_dict(torch.load(f = MODEL_LOAD_PATH), strict = False)
            print("WARNING: file loaded with strict = false, check before proceeding")
        except RuntimeError: 
            print("ERROR: unable to load file due to key missing / mismatch")

# compute the auroc of all 14 labels in one run, return the value, used for test set
# can be used ONLY if there are NO missing labels, i.e., the test set
def accuracy_auroc_all_labels(logits, tags):
    y_pred = torch.sigmoid(logits)
    auroc.update(y_pred, tags.int())
    return auroc.compute().item()

# restart the auroc list as defined in model cons
def auroc_list_reset():
    for aur in auroc_list:
        aur.reset()

# in the training cycle
def auroc_list_update(logits, tags):
    y_pred = torch.sigmoid(logits)
    for i, aur in enumerate(auroc_list):
        mask = ~torch.isnan(tags[:, i])
        if mask.sum() == 0:
            continue
        aur.update(y_pred[mask, i], tags[mask, i].int())

# compute the auroc value of 13 tags (excluding no findings)
def auroc_list_compute(verbose = True, debug = False):
    accuracy_list = np.zeros(14)
    for i, aur in enumerate(auroc_list):
        accuracy_list[i] = aur.compute().item()
    
    accuracy_list *= 100
    if debug:
        return accuracy_list

    no_finding_success = accuracy_list[0]
    avg_success = float(np.mean(accuracy_list[1:]))
    if verbose:
        high = np.argmax(accuracy_list[1:]) + 1
        low = np.argmin(accuracy_list[1:]) + 1
        print("average AUROC success is {:.2f}%".format(avg_success))
        print("{} has lowest success of {:.2f}%. {} has highest success of {:.2f}%. No finding success is {:.2f}%.".format(TAGS[low], accuracy_list[low], TAGS[high], accuracy_list[high], no_finding_success))
    return avg_success

# the accurary defined by labelling all of a patient's tags correctly, excluding no labels
def accuracy_strict(logits, tags):
    y_pred = torch.sigmoid(logits)
    mask = torch.isnan(tags)
    labels_pred = (y_pred > torch.tensor(THRESHOLDS).to(DEVICE)).int()
    labels_pred[mask] = 0
    tags = torch.nan_to_num(tags.type(torch.int8), nan = 0)
    corrects = (labels_pred[:, 1:] == tags[:, 1:]).all(dim = 1)
    return corrects.float().mean().item()

# training loop for one epoch only
def train_epoch(epoch = None):
    model.train()
    epoch_loss = 0.0
    if epoch is not None:
        print("Starting epoch {}".format(epoch))
    
    i = 0
    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        mask = ~torch.isnan(y_batch) # the places where the label exists
        y_batch = torch.nan_to_num(y_batch, nan = 0.0)

        y_logits = model(X_batch)
        loss_matrix = loss_fn(y_logits, y_batch) * mask
        loss_sum = loss_matrix.sum()
        loss = loss_sum / mask.sum()
        epoch_loss += loss_sum.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    
    if epoch is not None:
        print("Finished training epoch {} with loss {:2f} at time: {}".format(epoch, epoch_loss, time_passed()))
    return epoch_loss

# test a model
def test_model(debug = False, train_set = True, valid_set = True, test_set = False):
    model.eval()
    train_strict = 0.0
    valid_strict = 0.0
    train_auroc, valid_auroc, test_auroc = None, None, None

    with torch.inference_mode():
        if train_set:
            batches = len(train_dataloader)
            auroc_list_reset()
            for X_batch, y_batch in train_dataloader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                train_logits = model(X_batch)
                auroc_list_update(train_logits, y_batch)
                train_strict += accuracy_strict(train_logits, y_batch)
            train_strict /= batches
            print("Training set strict success {:.2f}%, ".format(train_strict * 100), end = "")
            train_auroc = auroc_list_compute(debug = debug)

        if valid_set:
            batches = len(valid_dataloader)
            auroc_list_reset()
            for X_batch, y_batch in valid_dataloader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                valid_logits = model(X_batch)
                auroc_list_update(valid_logits, y_batch)
                valid_strict += accuracy_strict(valid_logits, y_batch)
            valid_strict /= batches
            print("Validation set strict success {:.2f}%, ".format(valid_strict * 100), end = "")
            valid_auroc = auroc_list_compute(debug = debug)

        if test_set:
            test_images, test_tags = test.tensors
            test_images = test_images.to(DEVICE)
            test_tags = test_tags.to(DEVICE)
            test_logits = model(test_images)
            auroc_list_reset()
            auroc_list_update(test_logits, test_tags)
            test_strict = accuracy_strict(test_logits, test_tags)
            print("Test set strict success {:.2f}%, ".format(test_strict * 100), end = "")
            test_auroc = auroc_list_compute(debug = debug)
        
        print("Finished model testing at time: {}\n".format(time_passed()))
        return (train_auroc, valid_auroc, test_auroc)

# automatically train and test a model
def train_model(epochs = 10, start = 1):
    train_loss_list = []
    train_auroc_list = []
    valid_auroc_list = []

    if start > 1:
        print("Starting training from epoch {}".format(start))
        model = model_load(epoch = start - 1)
    else:
        test_model()

    for epoch in range(start, epochs + 1):
        try:
            train_loss_list.append(train_epoch(epoch))
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print("ERROR: CUDA out of memory!")
                torch.cuda.empty_cache()
            else:
                print("ERROR: {}".format(e))
            
            model_save(epoch)
            print("ERROR: Exiting training at time: {}\n".format(time_passed()))
            exit(1)

        if epoch % EVAL_FREQ == 0:
            train_auroc, valid_auroc, _ = test_model(test_set = False)
            train_auroc_list.append(train_auroc)
            valid_auroc_list.append(valid_auroc)
            scheduler.step(valid_auroc)

        if epoch % SAVE_FREQ == 0:
            model_save(epoch)
    
    print("Finished training {} epochs at time: {}".format(epochs, time_passed()))
    return(np.array(train_loss_list), np.array(train_auroc_list), np.array(valid_auroc_list))

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-lr", type = float, default = 1e-3)
    parser.add_argument("-wd", type = float, default = 1e-4)
    parser.add_argument("-epochs", type = int, default = 50)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    LR = args.lr
    WD = args.wd
    MAX_EPOCHS = args.epochs
    MODEL_NAME = "resnet_beta_LR_{}_WD_{}".format(LR, WD)
    SAVE_FREQ = MAX_EPOCHS
    setup()
    load_data(small = False)
    collect()
    model_cons()
    train_loss_list, train_auroc_list, valid_auroc_list = train_model(MAX_EPOCHS)
    print("\ntrain_loss: ", end = '')
    pprint(train_loss_list)
    print("train_auroc: ", end = '')
    pprint(train_auroc_list)
    print("valid_auroc: ", end = '')
    pprint(valid_auroc_list)
    test_model(train_set = False, valid_set = False, test_set = True)