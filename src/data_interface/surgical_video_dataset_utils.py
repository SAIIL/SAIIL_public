from collections import OrderedDict

import numpy as np
import torch


def make_numeric_pre_post(sample, past_emr_columns, future_emr_columns):
    """
    Convert fields to numeric values
    :param sample:
    :param past_emr_columns:
    :param future_emr_columns:
    :return:
    """
    # TODO(by guy.rosman): change fields into numerical values in a functor
    empty_symbol = ['','not listed']
    past_variable_dict = sample['past_variables'].copy()
    future_variable_dict = sample['future_variables'].copy()
    sample['past_variables'] = []
    sample['past_variables_valid'] = []
    sample['future_variables'] = []
    sample['future_variables_valid'] = []
    for key in past_emr_columns:
        if key not in past_variable_dict:
            if not key[0]=='_':
                sample['past_variables'].append(0.0)
            sample['past_variables_valid'].append(0.0)
        else:
            value = past_variable_dict[key]
            if value not in empty_symbol and value.isdigit():
                sample['past_variables'].append(float(value))
                sample['past_variables_valid'].append(1.0)
            elif value.lower().strip() == 'yes':
                sample['past_variables'].append(1.0)
                sample['past_variables_valid'].append(1.0)
            elif value.lower().strip() == 'no':
                sample['past_variables'].append(0.0)
                sample['past_variables_valid'].append(1.0)
            else:
                sample['past_variables'].append(0.0)
                sample['past_variables_valid'].append(1.0)

    for key in future_emr_columns:
        if key not in future_variable_dict:
            if not key[0]=='_':
                sample['future_variables'].append(0.0)
            sample['future_variables_valid'].append(0.0)
        else:
            value = future_variable_dict[key]
            if value not in empty_symbol and value.isdigit():
                sample['future_variables'].append(float(value))
                sample['future_variables_valid'].append(0.0)
            elif value.lower().strip() == 'yes':
                sample['future_variables'].append(1.0)
                sample['future_variables_valid'].append(0.0)
            elif value.lower().strip() == 'no':
                sample['future_variables'].append(0.0)
                sample['future_variables_valid'].append(0.0)
            else:
                sample['future_variables'].append(0.0)
                sample['future_variables_valid'].append(0.0)

    if len(sample['past_variables']) >0:
        sample['past_variables'] = np.array(sample['past_variables'])
    else:
        sample['past_variables'] = np.zeros((len(past_emr_columns),))

    if len(sample['future_variables']) >0:
        sample['future_variables'] = np.array(sample['future_variables'])
    else:
        sample['future_variables'] = np.zeros((len(future_emr_columns),))
    # import IPython;IPython.embed(header='use valid array')
    # TODO Yutong create onehot for categorical data and append in list for numerical data
    return sample


def merge_columns(row, column_names):
    """
    Transforms row, column names so as to merge multiple repeating columns. Prints out duplicate columns with mismatching data.
    (example transform -- others can be written and concatenated)
    :param row:
    :param column_names:
    :return:
    """
    result_dict = OrderedDict()
    for value, key in zip(row, column_names):
        if value.strip() == '':
            if key not in result_dict:
                result_dict[key] = value
        else:
            if key in result_dict and not result_dict[key].strip() == '' and not result_dict[
                                                                                     key].strip() == value.strip():
                # Check cases with more than one value that does not seem to match.
                # Keep the first non-zero entry, but report.
                print(row[0] + ', key: ' + key + ", value: " + str(value) + ', previous value: ' + result_dict[key])
            else:
                result_dict[key] = value
    result = list(result_dict.values())
    result_columns = list(result_dict.keys())
    return result, result_columns


def create_onehot(path, num_classes):
    batch_size = path.shape[0]
    time_size = path.shape[1]

    y_onehot = torch.zeros(batch_size, time_size,
                           num_classes).float().to(path.device)

    y_onehot.zero_()
    y_onehot.scatter_(2, path.long(), 1)
    return y_onehot


def create_onehot_nan(path, num_classes):
    batch_size = path.shape[0]
    time_size = path.shape[1]

    y_onehot = torch.zeros(batch_size, time_size,
                           num_classes).float().to(path.device)

    y_onehot.zero_()
    for batch_idx in range(batch_size):
        path_batch = path[batch_idx].squeeze(0).cpu().numpy()
        for idx, path_sample in enumerate(path_batch):
            if not np.isnan(path_sample):
                y_onehot[batch_idx, idx, path_sample] = 1.0
    return y_onehot