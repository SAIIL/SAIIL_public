import os
import torch
import numpy as np
import csv
import sys
import pickle
import torch.nn.functional as F
import pytorch_lightning as pl
from cv2 import log
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from model import CNN_LSTM
from data_interface.protobuf_dataset import process_data_directory_surgery
from misc.base_params import parse_arguments


# A pytorn script for model training and validation for phase recongition. 
# A script running example can be found in train_baseline.sh
# MIT liscence
# Author: Yutong Ban, Guy Rosman 

class TemporalTrainer(pl.LightningModule):

    def __init__(self, class_names = [], log_dir = './'):
        super().__init__()
        self.model = CNN_LSTM(n_class=len(class_names))
        self.class_names = class_names
        self.n_class = len(self.class_names)
        # create stat csv file to save all the intermidiate stats
        self.stat_file = open(os.path.join(log_dir, 'train_stats.csv'), 'w')
        self.stat_writer = csv.writer(self.stat_file)


    def on_train_start(self) -> None:
        stat_header = ['iter', 'metric']
        stat_header.extend(self.class_names)
        stat_header.append('mean')  
        self.stat_writer.writerow(stat_header)
        
        return super().on_train_start()


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = batch['imgs']
        y = batch['phase_trajectory']
        loss = 0
        y_hat = self.model(x)
        loss += F.cross_entropy(y_hat, y[:,-1, :].squeeze().argmax(dim=-1).type(torch.cuda.LongTensor))
        self.log('train_loss', loss)
        return loss

    def on_validation_start(self) -> None:
        # define cm
        self.cm = torch.zeros(self.n_class, self.n_class)

    def validation_step(self, batch, batch_idx):
        x = batch['imgs']
        y = batch['phase_trajectory']
        loss = 0
        y_hat = self.model(x)
        loss += F.cross_entropy(y_hat, y[:,-1, :].squeeze().argmax(dim=-1).type(torch.cuda.LongTensor))
        for idx_batch in range(y.shape[0]):
            gt = y[idx_batch, -1].argmax(dim=-1)
            est = y_hat[idx_batch].argmax(dim=-1)
            self.cm[int(gt.type_as(self.cm)), int(est.type_as(self.cm))] += 1.0
        self.log('val_loss', loss)
        return loss

    def on_validation_end(self) -> None:
        cm = self.cm.detach().cpu().numpy()
        accuracy = cm.diagonal() / cm.sum(axis=0)
        accuracy[np.isnan(accuracy)] = 0.0
        print("confusion matrix:")
        print(cm.astype(int))
        print("Recall:")
        accuracy = cm.diagonal() / cm.sum(axis=-1)
        accuracy[np.isnan(accuracy)] = 0.0
        stats = [self.current_epoch, "Recall"]
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(accuracy[idx]))
            stats.append(accuracy[idx])
        accuracy_mean = accuracy[accuracy != 0].mean()
        print('Overall recall' + ' :' + str(accuracy_mean) + '\n')
        stats.append(accuracy_mean)
        self.stat_writer.writerow(stats)

        print("Precision:")
        stats = [self.current_epoch, "Precision"]
        precision = cm.diagonal() / cm.sum(axis=0)
        precision[np.isnan(precision)] = 0.0
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(precision[idx]))
            stats.append(precision[idx])
        precision_mean = precision[precision != 0].mean()
        print('Overall precision' + ' :' + str(precision_mean)  + '\n')
        stats.append(precision_mean)
        self.stat_writer.writerow(stats)

    def on_test_start(self):
        self.gt = dict()
        self.est = dict()
        self.cm = torch.zeros(self.n_class, self.n_class)
        # create stat csv file to save all the inference stats
        self.test_stat_file = open(os.path.join(log_dir, 'test_stats.csv'), 'w')
        self.test_stat_writer = csv.writer(self.test_stat_file)
        stat_header = ['iter', 'metric']
        stat_header.extend(self.class_names) 
        stat_header.append('mean')
        self.test_stat_writer.writerow(stat_header)


    def test_step(self, batch, batch_idx):
        x = batch['imgs']
        y = batch['phase_trajectory']
        video_ids = batch['video_name']

        loss = 0
        y_hat = self.model(x)

        loss += F.cross_entropy(y_hat, y[:,-1, :].squeeze().argmax(dim=-1).type(torch.cuda.LongTensor))
        for idx_batch in range(y.shape[0]):
            video_id = video_ids[idx_batch]
            if video_id not in self.est.keys():
                self.est[video_id] = []
                self.gt[video_id] = []
            gt = y[idx_batch, -1].argmax(dim=-1)
            est = y_hat[idx_batch,:].argmax(dim=-1)
            self.cm[int(gt.type_as(self.cm)), int(est.type_as(self.cm))] += 1.0
            self.gt[video_id].append(int(gt))
            self.est[video_id].append(int(est))

        self.log('test_loss', loss)
        self.log('test_cm', self.cm)        
        return loss

    def on_test_end(self) -> None:
        cm = self.cm.detach().cpu().numpy()
        print("confusion matrix:")
        print(cm.astype(int))
        print("Recall:")
        stats = [self.current_epoch, "Recall"]
        accuracy = cm.diagonal() / cm.sum(axis=-1)
        accuracy[np.isnan(accuracy)] = 0.0
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(accuracy[idx]))
            stats.append(accuracy[idx])
        accuracy_mean = accuracy[accuracy != 0].mean()
        print('Overall recall' + ' :' + str(accuracy_mean) + '\n')
        stats.append(accuracy_mean)
        self.test_stat_writer.writerow(stats)

        print("Precision:")
        stats = [self.current_epoch, "Precision"]
        precision = cm.diagonal() / cm.sum(axis=0)
        precision[np.isnan(precision)] = 0.0
        for idx, class_name in  enumerate(self.class_names):
            print(class_name + ':' + str(precision[idx]))
            stats.append(precision[idx])
        precision_mean = precision[precision != 0].mean()
        print('Overall precision' + ' :' + str(precision_mean)  + '\n')
        stats.append(precision_mean)
        self.test_stat_writer.writerow(stats)

        return {'gt':self.gt, 'est':self.est}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def additional_arg_setter(parser):
    parser.add_argument('--gpu', action='append', type=int, default=[], help="")
    parser.add_argument('--exp_name', action='store', type=str, default='sleeve', help="")
    parser.add_argument('--train', default=False, action='store_true',  help="train model")
    parser.add_argument('--load_checkpoint', default=False, action='store_true',  help="train model")
    parser.add_argument('--checkpoint_path', action='store', type=str, default=None, help="the path to load the checkpoints")
    parser.add_argument('--inference', default=False, action='store_true', help="run model")
    parser.add_argument('--load_inference_result', default=False, action='store_true', help="load inference results")
    return parser

if __name__ == "__main__":
    # A pytorn script for model training and validation for phase recongition. 
    # a script example is in train_model.sh
    ### python train_baseline.py --track_name phase --data_dir DATA_PATH/videos/
    # --annotation_filename DATA_PATH/annotations/ --temporal_length 8 --sampling_rate 1 --cache_dir ./cache --num_dataloader_workers 8 --num_epochs 20

    args = parse_arguments(additional_setters=[additional_arg_setter])
    params = vars(args)

    dataset_splits = ['train','test']
    training_ratio = {'train':1.0,
                      'test': 0.0}

    log_dir = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    for split in dataset_splits:
        datasets = process_data_directory_surgery(
            data_dir=args.data_dir,
            fractions=args.fractions,
            width=args.image_width,
            height=args.image_height,
            sampling_rate=args.sampling_rate,
            past_length=args.temporal_length,
            batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers,
            sampler=None,
            verbose=False,
            annotation_folder=args.annotation_filename + '/' +split,
            temporal_len=args.temporal_length,
            train_ratio=training_ratio[split],
            skip_nan=True,
            seed=1234,
            phase_translation_file=args.phase_translation_file,
            cache_dir=args.cache_dir,
            params=params,
        )

        if split == 'train':
            train = datasets["train"]
        elif split == 'test':
            val = datasets["val"]

    dataloader_train = DataLoader(train, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                  num_workers=args.num_dataloader_workers)
    dataloader_test = DataLoader(val, batch_size=args.batch_size, drop_last=True,
                                 num_workers=args.num_dataloader_workers)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name='lightning')

    model = TemporalTrainer(class_names=train.class_names, log_dir = log_dir)

    trainer = pl.Trainer(gpus=args.gpu, accelerator='ddp', check_val_every_n_epoch=1, max_epochs=args.num_epochs, logger=tb_logger)
    # trainer = pl.Trainer(gpus=1, check_val_every_n_epoch=1, max_epochs=args.num_epochs, logger=tb_logger)

    if params['load_checkpoint']:
        if params['checkpoint_path'] == None: 
            params['checkpoint_path'] =  'default path'
        model = model.load_from_checkpoint(
                    checkpoint_path=params['checkpoint_path'], n_class=len(train.class_names))
        print("Checkpoint loaded")

    if params['train']:
        trainer.fit(model, dataloader_train, dataloader_test)
    
    
    if params['load_inference_result']:
        with open(os.path.join(log_dir, params['exp_name'] + '.pkl'), 'rb') as f:
            p_data = pickle.load(f)
            model.est = p_data['est']
            model.gt = p_data['gt']

    elif params['inference']:
        stats = trainer.test(model, dataloaders = dataloader_test)

        save_dict = {'gt':model.gt, 'est':model.est}
        with open(os.path.join(log_dir, params['exp_name'] + '.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)
