import copy
from collections import OrderedDict

import torch
import torchvision
from phase_net.temporal_model_interface import TemporalModelItfc
from torch import nn


def create_fc_net(sizes):
    module_list=[]
    module_list.append(torch.nn.Dropout())
    for i,(sz_a, sz_b) in enumerate(zip(sizes[:-1],sizes[1:])):
        module_list.append(torch.nn.Linear(sz_a,sz_b))
        if (i<(len(sizes)-2)):
            module_list.append(torch.nn.BatchNorm1d(sz_b))
            module_list.append(torch.nn.ReLU())
            module_list.append(torch.nn.Dropout())

    return torch.nn.Sequential(*module_list)


class VisualModel(torch.nn.Module):
    def __init__(self,base_model,num_phases):
        super().__init__()
        self.base_model=base_model
        self.base_model_output_dim = self.base_model.fc[-1].out_features
        self.num_phases=num_phases
        self.phases_head=create_fc_net([self.base_model_output_dim, num_phases])
        self.auxiliary_task_heads=torch.nn.modules.container.ModuleDict()
        self.fc=base_model.fc

    def get_number_of_phases(self):
        return self.num_phases

    def get_latent_size(self):
        return self.base_model_output_dim

    def get_phases_head(self):
        return self.phases_head

    def forward_latent(self,input):
        output=self.base_model(input)
        return output

    def forward(self,input):
        """
        The forward function for the main task -- phases in our case
        :param input:
        :return:
        """
        output=self.base_model(input)
        phases_output=self.phases_head(output)
        return phases_output

def create_simple_visual_network(num_classes, int_dim = [200,100]):
    '''
    Creates a visual classification network based on a resnet structure
    :param num_classes: Number of classes to be used..
    :param int_dim: FC layers to use.
    :return: a torch module that accepts images and returns the scores of each of the num_classes classes
    '''
    model = torchvision.models.resnet18(pretrained=True)

    sizes=[model.fc.in_features]
    sizes.extend(int_dim)
    # sizes.append(num_classes)
    module_list=[]
    module_list.append(torch.nn.Dropout())
    for i,(sz_a, sz_b) in enumerate(zip(sizes[:-1],sizes[1:])):
        module_list.append(torch.nn.Linear(sz_a,sz_b))
        if (i<(len(sizes)-2)):
            module_list.append(torch.nn.BatchNorm1d(sz_b))
            module_list.append(torch.nn.ReLU())
            module_list.append(torch.nn.Dropout())

    model.fc = torch.nn.Sequential(*module_list)
    multitask_model= VisualModel(model, num_phases=num_classes)

    for param in multitask_model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in multitask_model.phases_head.parameters():
        param.requires_grad = True

    return multitask_model


class TemporalModel(TemporalModelItfc):
    """
    Vanilla temporal model w/ LSTM
    """
    def __init__(self,
                 num_classes,
                 device,
                 lstm_size,
                 interface_size=None,
                 params=dict()):
        super().__init__()
        self.analysis_modes = params['model_analysis_modes']
        self.device = device
        self.batch_first = True
        self.params = params
        self.lstm_size = lstm_size
        self.interface_size = interface_size
        self.num_classes = num_classes
        self.tbptt = params.get('tbptt', 0)

        # base model
        visual_embedding_length = params.get('visual_embedding_length', 256)
        self.visual_encoder = self.create_visual_encoder(num_classes=num_classes,
                                                         interface_size=visual_embedding_length,
                                                         remove_layers=4,
                                                         device=device)

        # the visual model for generating the visual-only inference
        self.visual_model = copy.deepcopy(self.visual_encoder)

        # accepts video images, estimates lstm_size state space
        self.generator_encoder = nn.LSTM(
            input_size=visual_embedding_length + interface_size,
            hidden_size=lstm_size,
            batch_first=self.batch_first)

        # accepts interface_size state space, generates phase distribution
        self.phase_module = nn.Linear(in_features=lstm_size,
                                      out_features=num_classes)

        self.discrete_softmax_coeff = params.get('discrete_softmax_coeff', 1)
        self.softmax_layer = torch.nn.Softmax(dim=-1)

        if not self.device is None:
            self.generator_encoder = self.generator_encoder.to(device)
            self.phase_module = self.phase_module.to(device)


    def create_visual_encoder(self, num_classes=None,
                              remove_layers=4,
                              interface_size=256,
                              device=None):

        visual_encoder = create_simple_visual_network(num_classes=num_classes,
                                               int_dim=[interface_size * 2, interface_size])

        updated_visual_model = copy.deepcopy(visual_encoder)

        return updated_visual_model

    def get_num_phases(self):
        return self.num_classes


    def get_phase_identification_params(self, requires_grad=True):
        """
        Get the parameters for phase identification module: visual encoder + temporal model
        :param requires_grad: if True, return only parameters that have requires_grad
        :return:
        """
        params = []

        for p in self.generator_encoder.parameters():
            if p.requires_grad or not requires_grad:
                params.append(p)

        for p in self.visual_encoder.parameters():
            if p.requires_grad or not requires_grad:
                params.append(p)

        for p in self.phase_module.parameters():
            if p.requires_grad or not requires_grad:
                params.append(p)

        return params

    def forward(self, x_in, h_in):
        pass

    def init_hidden_state(self, example_tensor):
        pass

    def predict(self, observations, start_i=0):
        pass

    def predict_log_prob(self, observations, start_i=None):
        return None

    def generate_past_phase_belief(self, observations, imgs):
        """
        generate a sample path, given as class belief per timepoint
        :param observations:
        :param imgs: the imgs input
        :param path_len:
        :param start_i: the index to start from when making predictions. i=0 means the beginning of the observations (i.e. estimation, not prediction).
        :return: phase belief vector, global parameters belief vector
        """
        if 'phase_identification' not in self.analysis_modes:
            assert ('The current working mode does not support the past phase identifications!')

        visual_embedding = []
        for t in range(imgs.shape[1]):
            visual_embedding.append(
                self.visual_encoder.forward_latent(imgs[:, t, :, :, :].to(self.device)).unsqueeze(1))
        visual_embedding = torch.cat(visual_embedding, dim=1)

        # TODO(guy.rosman): remove the use of observations, both here and in the interface.
        zeros_tmp = torch.zeros_like(observations)

        concatenated_embedding = torch.cat([visual_embedding, zeros_tmp], dim=2)

        h_state = torch.zeros(1, concatenated_embedding.shape[0], self.lstm_size).to(self.device)
        c_state = torch.zeros(1, concatenated_embedding.shape[0], self.lstm_size).to(self.device)

        encoded_phase = []
        for t in range(imgs.shape[1]):
            res_encoder, (h_state, c_state) = self.generator_encoder(concatenated_embedding[:, t, :].unsqueeze(1),
                                                                     (h_state, c_state))
            encoder_generated = self.phase_module(res_encoder.squeeze(2)).squeeze(0)
            encoded_phase.append(encoder_generated)
            # truncated back probagation through time
            if (self.tbptt is not None) and (imgs.shape[1] - self.tbptt == 0):
                h_state = h_state.detach()
                c_state = c_state.detach()

        encoded_phase = torch.cat(encoded_phase, dim=1)
        # TODO(guy.rosman): consider feeding in a sample of the phase selected?

        return encoded_phase, None

    def generate_phase_visual_model(self, imgs):
        """
        Generate a sample path, given as class belief per timepoint
        :param imgs: the imgs input
        :return:
        """
        visual_prob = []
        for t in range(imgs.shape[1]):
            prob = self.visual_encoder(imgs[:, t, :, :, :].to(self.device))
            visual_prob.append(prob.unsqueeze(1))
        visual_prob = torch.cat(visual_prob, dim=1)

        return visual_prob