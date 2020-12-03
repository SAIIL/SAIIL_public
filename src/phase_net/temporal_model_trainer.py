import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from data_interface.surgical_dataset import DATASET_KEY_IMAGES


class TemporalTrainer:
    def __init__(self, model, data, params, device, logger):
        """

        :param model: the temporal model to be trained
        :param params: a dictionary with all parameters for the training
        :param device:
        """
        self.model = model
        self.params = params
        self.learning_rate = params["learning_rate"]
        self.phase_id_params = self.model.get_phase_identification_params()
        self.phase_identification_optimizer = torch.optim.Adam(
            params=self.phase_id_params, lr=self.learning_rate, weight_decay=1e-4
        )
        self.device = device
        self.epoch_size_phase_id = params["epoch_size_phase_id"]
        self.path_length = params["temporal_length"]
        self.mse_coeff = params.get("mse_coeff", "0.0")
        self.logger = logger
        self.num_classes = model.get_num_phases()
        self.log_prefix = params.get("log_prefix", "000")
        self.total_batch_cnt = 0
        self.partial_data = None
        self.update_real_data(data)
        self.tbptt = params.get("tbptt",'5')
        self.data_loss = torch.nn.CrossEntropyLoss(weight=self.real_dataset.dataset.class_weights.float().to(device))
        self.data_loss_past = torch.nn.CrossEntropyLoss(
            weight=self.real_dataset.dataset.class_weights.float().to(device)
        )
        self.dropout_ratio = params.get("dropout_ratio", 0.3)
        self.phase_count = np.zeros(self.num_classes)
        self.n_phase_count = np.zeros(self.num_classes)
        self.input_surgeon_id = params.get("input_surgeon_id", None)

    def update_real_data(self, data):
        """
        :param data:  a dataset with (input, future trajectory_sample) per item
        :return:
        """
        self.real_dataset = data

    def reshuffle_data(self, epoch_size=128):
        indices = np.random.choice(
            a=len(self.real_dataset.dataset), size=epoch_size, p=self.real_dataset.dataset.segment_weights_list
        )

        self.partial_data = DataLoader(
            self.real_dataset.dataset,
            batch_size=self.real_dataset.batch_size,
            sampler=SubsetRandomSampler(indices),
            num_workers=self.real_dataset.num_workers,
        )

    def train_phase_identification(self, pretrain_flag="visual"):
        """
        Train phase identification only, for pretraining.
        """
        self.reshuffle_data(epoch_size=self.epoch_size_phase_id)
        mses = []
        total_cost = 0.0
        total_data_loss_past = 0.0
        p1 = self.model.get_phase_identification_params()
        p1 = torch.cat([x.flatten() for x in p1]).clone().detach()
        for idx, sample in enumerate(self.partial_data):
            inps = sample["phase_trajectory"]
            inps_img = sample[DATASET_KEY_IMAGES]
            inps = inps.to(self.device)

            self.phase_identification_optimizer.zero_grad()

            cost, stats = self.phase_identification_cost(inps, inps_img, pretrain_flag)
            cost.backward()

            self.phase_identification_optimizer.step()

            total_cost += cost.clone().detach().cpu().item()
            total_data_loss_past += stats["data_loss_past"]

        p2 = self.model.get_phase_identification_params()
        p2 = torch.cat([x.flatten() for x in p2]).clone().detach()

        if self.logger:
            scalars = {
                "data_cost_past": total_data_loss_past / idx,
                "residual": ((p1 - p2) ** 2).mean().cpu().item(),
                "log10_residual_generator": ((p1 - p2) ** 2 + 1e-10).mean().log10().cpu().item(),
            }
            self.logger.log_train(self.total_batch_cnt, scalars, None, prefix=self.log_prefix)

    def phase_identification_cost(self, inps, imgs, pretrain_flag="lstm"):
        stats = {}
        if pretrain_flag == "lstm":
            past_sample_prob, _ = self.model.generate_past_phase_belief(inps, imgs)
        elif pretrain_flag == "visual":
            past_sample_prob = self.model.generate_phase_visual_model(imgs)

        cost = 0
        labels = inps[:, :, : self.num_classes].argmax(dim=2)

        seq_length = past_sample_prob.shape[1]
        for t in range(past_sample_prob.shape[1]):
            cost += self.data_loss_past(past_sample_prob[:, t, :], labels[:, t].squeeze().clone().detach().long())

        stats["data_loss_past"] = cost.clone().detach().cpu().item() / seq_length
        self.total_batch_cnt += 1
        if self.params["verbose"]:
            print("cost:" + str(cost))
            print("")
        return cost, stats
