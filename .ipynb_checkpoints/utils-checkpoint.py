import os
import glob
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import io
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


# LABEL2NUM = {
#     'shallow': 0,
#     'medium': 1,
#     'deep' : 2
# }

# NUM2LABEL = {
#     0: 'shallow',
#     1: 'medium',
#     2: 'deep'
# }
label2num = {x:idx for idx, x in enumerate(os.listdir('/content/drive/MyDrive/ĐATN20231/vn/cat_images'))}
num2label = {v:k for k, v in label2num.items()}



def split_train_test_data(dataset_dir='/content/drive/MyDrive/ĐATN20231/vn/cat_images'):
    """
        directory sample: ./dent_data/deep/9_0.png
    """
    label2num = {x:idx for idx, x in enumerate(os.listdir(dataset_dir))}
    num2label = {v:k for k, v in label2num.items()}
    keys = list(label2num.keys())
    all_data = []
    for k in keys:
        dirs = [os.path.join(f'{dataset_dir}/{k}/', f'{x}') for x in os.listdir(f'{dataset_dir}/{k}')]
        all_data += dirs
    # shallow_image_dir = glob.glob(dataset_dir + 'shallow/' + '*.png')
    # medium_image_dir = glob.glob(dataset_dir + 'medium/' + '*.png')
    # deep_image_dir = glob.glob(dataset_dir + 'deep/' + '*.png')
    # all_data = shallow_image_dir + medium_image_dir + deep_image_dir
    levels = [x.split('/')[-2] for x in all_data]
    labels = [label2num[i] for i in levels]
    train_x, val_x, train_y, val_y = train_test_split(all_data, labels, test_size=0.2, random_state=42)

    return train_x, val_x, train_y, val_y


def make_data_classification_from_yolo_format(
    obj_name_file_dir = './dataset/obj.names', 
    data_dir='./dataset/img/', 
    output_dir='./dent_data/'):
    
    obj_file = open(obj_name_file_dir)
    class_names = obj_file.read()
    class_names = class_names.splitlines()
    class_nums = ['0.0','1.0','2.0']
    num2label = dict(zip(class_nums, class_names))
    image_files = glob.glob(data_dir + '*.jpeg')
    
    print(num2label)

    for c in class_names:
        directory = output_dir + c
        if not os.path.exists(directory):
            os.makedirs(directory)

    for img_file in tqdm(image_files):
        txt_dir = img_file[:-5] + '.txt'
        txt_file = open(txt_dir)
        ann = txt_file.read()
        ann = ann.splitlines()
        img = cv2.imread(img_file)
        dh, dw, _ = img.shape
        name = img_file.split('/')[-1][:-5]

        for idx, bbox in enumerate(ann):
            c, x_center, y_center, w, h = map(float, bbox.split(' '))

            x1 = int((x_center - w / 2) * dw)
            y1 = int((y_center - h / 2) * dh)
            x2 = int((x_center + w / 2) * dw)
            y2 = int((y_center + h / 2) * dh)

            if x1 < 0:
                x1 = 0
            if x2 > dw - 1:
                x2 = dw - 1
            if y1 < 0:
                y1 = 0
            if y2 > dh - 1:
                y2 = dh - 1
            
            cv2.imwrite(f'{output_dir}{num2label[str(c)]}/{name}_{idx}.png', img[y1:y2, x1:x2])

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

if __name__ == "__main__":
    # with open('dataset_24_09_2023.pkl', 'wb') as f:
    #     pickle.dump(split_train_test_data(), f)
    # print(label2num)
    with open('dataset_24_09_2023.pkl', 'rb') as file:
        train_x, val_x, train_y, val_y = pickle.load(file)
  
    print(f'{len(train_x)} - {len(val_x)}')
    print(train_x[0])


