import os

try:
    import tensorflow
except ImportError:
    print("No tensorflow available")

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("No tensorboardX   available")


class TrainingLogger:
    #  logger based on tensorflow, for the train and test phases.

    def __init__(self, exp_name="", log_dir="logs"):
        # exp_name - the name of the experiment
        self.train_logger = SummaryWriter(os.path.join(log_dir, exp_name), flush_secs=5)
        self.test_logger = self.train_logger

    def log_train(self, step, scalars, weights, prefix=None, text={}, fig={}):
        """

        :param step: the step/iteration
        :param scalars: a dictionary of name -> value
        :param weights: for histograms, TBD
        :param prefix: prefix name for the training run
        :param text: a dictionary of name -> text
        :return:
        """
        str_prefix = "" if prefix is None else prefix + "_"
        for x in scalars.keys():
            self.train_logger.add_scalar(tag=str_prefix + "train/" + str(x), scalar_value=scalars[x], global_step=step)
        for x in text.keys():
            self.train_logger.add_text(tag=str_prefix + "train/" + str(x), text_string=text[x], global_step=step)
        if weights:
            for x in weights.keys():
                self.train_logger.add_histogram(
                    tag=str_prefix + "train/" + str(x),
                    bins=weights[x]["bins"],
                    values=weights[x]["values"],
                    global_step=step,
                )
        for x in fig.keys():
            self.train_logger.add_figure(tag=str_prefix + "train/" + str(x), figure=fig[x], global_step=step)

        self.train_logger.close()

        # TODO: log text

    def log_test(self, step, scalars, weights, prefix=None, text={}, fig={}):
        str_prefix = "" if prefix is None else prefix + "_"
        for x in scalars.keys():
            self.test_logger.add_scalar(tag=str_prefix + "test/" + str(x), scalar_value=scalars[x], global_step=step)
        for x in text.keys():
            self.test_logger.add_text(tag=str_prefix + "test/" + str(x), text_string=text[x], global_step=step)
        if weights:
            for x in weights.keys():
                self.test_logger.add_histogram(
                    tag=str_prefix + "test/" + str(x),
                    bins=weights[x]["bins"],
                    values=weights[x]["values"],
                    global_step=step,
                )
        for x in fig.keys():
            self.test_logger.add_figure(tag=str_prefix + "train/" + str(x), figure=fig[x], global_step=step)

        self.test_logger.close()

        # TODO: log text
