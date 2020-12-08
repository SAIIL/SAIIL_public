import copy
import hashlib
import os
import time

import torch.nn
import tqdm
from torch.utils.data import DataLoader
from misc.base_params import parse_arguments
from data_interface.base_utils import collate_filter_empty_elements
from data_interface.protobuf_dataset import process_data_directory_surgery
from misc.training_logger import TrainingLogger
from phase_net.discrete_temporal_model import TemporalModel
from phase_net.temporal_model_trainer import TemporalTrainer

if __name__ == "__main__":
    """
    A training example for phase segementation network.
    """
    args = parse_arguments()
    simulate_data = False
    torch.multiprocessing.set_sharing_strategy("file_system")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    # pass cuda device number via  args.cuda_device, then cuda:0 represents to use this device
    if args.cuda_device == "cpu":
        str_cuda_device = "cpu"
    else:
        str_cuda_device = "cuda:0"
    device = torch.device(str_cuda_device if (torch.cuda.is_available() and not args.disable_cuda) else "cpu")
    print("device = cuda:" + str(device))

    save_step = 10

    params = vars(args)

    trajectory_prediction_datasets = process_data_directory_surgery(
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
        annotation_filename=args.annotation_filename,
        temporal_len=args.temporal_length,
        train_ratio=args.training_ratio,
        skip_nan=True,
        seed=1234,
        phase_translation_file=args.phase_translation_file,
        cache_dir=args.cache_dir,
        params=params,
    )

    dataset = trajectory_prediction_datasets["train"]
    phases = len(dataset.class_names)
    interface_size = phases
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers,
        collate_fn=collate_filter_empty_elements,
    )

    temporal_model = TemporalModel(
        num_classes=phases, device=device, lstm_size=32, interface_size=interface_size, params=params
    )

    device = torch.device("cpu")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        temporal_model = temporal_model.cuda()

    hash = hashlib.sha1()
    hash.update(str(time.time()).encode("utf-8"))
    hash_str = str(hash.hexdigest())

    logger = TrainingLogger(exp_name="test_gan_" + hash_str, log_dir=args.log_dir)

    trainer = TemporalTrainer(temporal_model, params=params, data=dataloader, device=device, logger=logger)

    # train all the parameters
    for param in temporal_model.visual_encoder.parameters():
        param.requires_grad = True

    # pre_train data loss past
    statedict_filename = args.model_filename
    fullmodel_filename = statedict_filename
    for it in tqdm.tqdm(range(args.num_epochs), desc="phase identification iterations"):
        if it < args.num_epochs / 2:
            pretrain_flag = "visual"  
        else:
            pretrain_flag = "lstm"
        if it == args.num_epochs / 2:
            trainer.model.visual_model = copy.deepcopy(trainer.model.visual_encoder)
            trainer.phase_identification_optimizer = torch.optim.Adam(
                params=trainer.phase_id_params, lr=0.1 * trainer.learning_rate, weight_decay=1e-4
            )
        trainer.train_phase_identification(pretrain_flag)

        if it % save_step == 0 and it > 0:
            torch.save(trainer.model, fullmodel_filename)
    print("Finish model training")
