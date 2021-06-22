from abc import abstractmethod

from torch import nn


class TemporalModelPublicItfc(nn.Module):
    """Interface for a temporal model (abstract class)"""

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x_in, h_in):
        pass

    @abstractmethod
    def init_hidden_state(self, example_tensor):
        pass

    # TODO(guy.rosman) CLEANUP_10_9 remove observations from signature.
    @abstractmethod
    def generate_past_phase_belief(self, observations, imgs, tbptt_burnin=None):
        pass

    @abstractmethod
    def predict(self, observations, start_i=0):
        pass

    @abstractmethod
    def predict_log_prob(self, observations, start_i=0, tbptt_burnin=None):
        pass


class TemporalModelItfc(TemporalModelPublicItfc):
    """Interface for a temporal model (abstract class)"""

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x_in, h_in):
        pass

    @abstractmethod
    def init_hidden_state(self, example_tensor):
        pass

    @abstractmethod
    def generate_past_phase_belief(self, observations, imgs, tbptt_burnin=None):
        """

        :param observations:
        :param imgs:
        :param tbptt_burnin:
        :return:
        """
        pass

    @abstractmethod
    def get_phase_identification_params(self, requires_grad=True):
        """

        :param requires_grad:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, observations, start_i=0):
        pass

    @abstractmethod
    def predict_log_prob(self, observations, start_i=0, tbptt_burnin=None):
        """
        Compute the log-probabilities of the current step, posterior estimate, phases.
        :param observations:
        :param start_i:
        :param tbptt_burnin:
        :return:
        past_encoded_results
        result - dictionary: channel name -> one- hot distribution, or parameters of that channel
        predicted_parameters - dictionary: additional parameter names -> posterior estimates of global parameters (e.g not per time frame).
        """
        pass

    @abstractmethod
    def add_task(self, task_name, task_module, task_type):
        """

        :param task_name:
        :param task_module:
        :param task_type: Specifies whether the task is global (in time) / local (per time frame)
        :return:
        """
        pass

    @abstractmethod
    def forward_task(self, task_name):
        """
        Run the task module
        :param task_name:
        :return:
        """
        pass

    @abstractmethod
    def get_num_phases(self):
        """
        Return the number of phases
        :return:
        """
        pass

    @abstractmethod
    def generate_past_phase_belief(self, observations, imgs):
        """
        generate a sample path, given as class belief per timepoint
        :param observations:
        :param imgs: the imgs input
        :return: phase belief vector, global parameters belief vector
        """
