import argparse


def parse_arguments(additional_setters=[],):
    """load and parse arguments

    Args:
        additional_setters (list, optional): List of additional parameter setter functions of the form setter(parser). Defaults to [].

    Returns:
        Arguments object: set of arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    parser.add_argument("--show_samples", action="store_true", help="")
    parser.add_argument("--data_dir", action="store", default="~/data_mgh/AI Sleeve Videos/", help="Data folder")
    parser.add_argument("--log_dir", action="store", default="./logs/", help="Data folder")
    parser.add_argument("--cache_dir", action="store", default="~/cache_mgh", help="Cache dir for datasets")

    parser.add_argument("--annotation_filename", action="store", default="Updated Annotations Sleeve.ods", help="")
    parser.add_argument("--results_filename", action="store", default="results.pkl", help="")
    parser.add_argument("--model_filename", action="store", default="default_modelname", help="")
    parser.add_argument("--temporal_model_filename", action="store", default="default_temporal_modelname", help="")
    parser.add_argument("--output_filename", action="store", default="output.pkl", help="")
    parser.add_argument("--num_epochs", action="store", type=int, default=20, help="")
    parser.add_argument("--learning_rate", action="store", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument(
        "--fractions",
        action="store",
        default=1.0,
        help="The amount of samples to take from the data, as a float, or a point to a json that matches phase to fraction.",
    )
    parser.add_argument("--min_seg_fraction", action="store", type=float, default=0.99999, help="")
    parser.add_argument("--max_seg_fraction", action="store", type=float, default=1.0, help="")
    parser.add_argument("--training_ratio", action="store", type=float, default=0.7, help="")
    parser.add_argument("--segment_ratio", action="store", type=float, default=1.0, help="")
    parser.add_argument("--sampling_step", action="store", type=int, default=50, help="")
    parser.add_argument("--temporal_length", action="store", type=int, default=8, help="")
    parser.add_argument("--lstm_size", action="store", type=int, default=64, help="")
    parser.add_argument("--num_dataloader_workers", action="store", type=int, default=4, help="")
    parser.add_argument("--saving_step", action="store", type=int, default=10, help="")
    parser.add_argument("--batch_size", action="store", type=int, default=64, help="")
    parser.add_argument("--image_width", action="store", type=int, default=224, help="")
    parser.add_argument("--image_height", action="store", type=int, default=224, help="")
    parser.add_argument("--epoch_size", action="store", type=int, default=2048, help="epoch size")
    parser.add_argument("--epoch_size_phase_id", action="store", type=int, default=10240, help="epoch size")
    parser.add_argument(
        "--num_training_experiments",
        action="store",
        type=int,
        default=1,
        help="number of training experiments/repetitions",
    )
    parser.add_argument("--disable_cuda", action="store", type=bool, default=False, help="should disable cuda")
    parser.add_argument("--cuda_device", action="store", type=str, default="0", help="cuda device index: 0 or 1")
    parser.add_argument("--track_name", action="store", type=str, default=None, help="track name to use for phases")
    # JSON file to map phase names
    parser.add_argument("--phase_translation_file", action="store", default=None, help="")

    parser.add_argument(
        "--view_examples",
        action="store",
        type=str,
        default="all",
        help="A str of (gt_label)_(pred_label) denoting which examples to view, e.g. '1,2'. If set to 'errors', view all examples that are mislabeled in the confusion matrix. If 'all', sample all examples (need to add)",
    )
    parser.add_argument(
        "--sampling_rate", action="store", type=float, default=5, help=""
    )  # unit frame per second (fps)
    parser.add_argument(
        "--phase_order_filename",
        action="store",
        default=None,
        help="Phase order json, to make visualization well defined.",
    )
    parser.add_argument("--phase_transition_filename", action="store", default=None, help="")
    parser.add_argument("--video_name_list_filename", action="store", default=None, help="")
    parser.add_argument("--training_filename", action="store", default=None, help="")
    parser.add_argument("--inference_filename", action="store", default=None, help="")
    parser.add_argument(
        "--multitask_list",
        "--list",
        action="append",
        default=[],
        help="e.g. progress_regressor or img_reconstruction",
        required=False,
    )
    parser.add_argument("--plot_fingerprints", action="store", type=bool, default=False, help="should plot results")
    parser.add_argument(
        "--plot_confusion_matrix", action="store", type=bool, default=False, help="should plot confusion matrix"
    )
    parser.add_argument("--plot_video", action="store", type=bool, default=False, help="should plot video")
    parser.add_argument("--write_txt_results", action="store", type=bool, default=True, help="should plot results")
    parser.add_argument(
        "--skip_full_forward", action="store_true", help="Do not run the full forward inference, use zeros"
    )
    parser.add_argument("--mse_coeff", action="store", type=float, default=0.5, help="")
    parser.add_argument("--verbose", action="store_true", default=None, help="")

    parser.add_argument("--phase_pretrain_iter", action="store", type=int, default=100, help="")



    parser.add_argument(
        "--model_analysis_modes",
        nargs="+",
        default=["phase_identification"],
        help="Define which modes of analysis are available by the model",
    )

    if additional_setters is not None:
        if type(additional_setters) is not list:
            additional_setters = [additional_setters]
        for setter in additional_setters:
            parser = setter(parser)

    args = parser.parse_args()
    return args
