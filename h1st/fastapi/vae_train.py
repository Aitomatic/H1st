import os
import sys

from vae import PredictiveMaintenanceVAE


def main(train_param, tfr_info):
    pm_vae = PredictiveMaintenanceVAE(
        input_dim=tfr_info['columns'],
        n_window=train_param['n_window'],
        h_dim=train_param['h_dim'],
        z_dim=train_param['z_dim'],
        n_layer=train_param['n_layer'],
        batch_norm=train_param['batch_norm'],
        alpha=train_param['alpha'],
        n_rank=train_param['n_rank'])

    pm_vae.train(
        input_path=tfr_info['tfr_path'],
        save_path=train_param['save_model_path'],
        tfr_id_key=tfr_info['column_names']['id_key'],
        tfr_datetime_key=tfr_info['column_names']['datetime_key'],
        tfr_feature_key=tfr_info['column_names']['feature_key'],
        pre_fix=tfr_info['tfr_file_prefix'],
        data_cache=train_param['data_cache'],

        n_feature=tfr_info['columns'],
        n_window=train_param['n_window'],
        n_row=tfr_info['rows'],
        batch_size=train_param['batch_size'],

        interval=tfr_info['interval'],

        num_epochs=train_param['num_epochs'],

        beta=train_param['beta'],

        learning_rate_init=train_param['learning_rate_init'],
        learning_rate_decay_epoch=train_param['learning_rate_decay_epoch'],
        learning_rate_decay_rate=train_param['learning_rate_decay_rate'],
        max_to_keep=train_param['max_to_keep'],
        model_saving_interval=train_param['model_saving_interval'],
        summary_step=train_param['summary_step'],
        printing_step=train_param['printing_step'])


TFRECORDS_PREFIX = os.environ["TFRECORDS_PREFIX"]
CHECKPOINT_PREFIX = os.environ["CHECKPOINT_PREFIX"]

N_COLS = {
    'group_1': 6,
    'group_2': 7,
    'group_5': 7,
}

BATCH_SIZES = {
    'group_1': 512,
    'group_2': 32,
    'group_5': 32,
}

if __name__ == "__main__":
    sensor_group_name = sys.argv[1]
    operation_mode = 'Cooling'
    tfrecords_prefix = "%s.all/%s.tfrecords/operation_mode=%s" % (TFRECORDS_PREFIX, sensor_group_name, operation_mode)
    checkpoint_prefix = "%s/%s/operation_mode=%s/" % (CHECKPOINT_PREFIX, sensor_group_name, operation_mode)

    train_param = {
        "data_cache": True,
        "z_dim": 3,
        "h_dim": 256,
        "n_window": 1,
        "n_layer": 3,
        "batch_norm": True,
        "alpha": 0.1,
        "beta": 100,
        "n_rank": 3,
        "batch_size": BATCH_SIZES[sensor_group_name],
        "num_epochs": 50,
        "learning_rate_init": 0.00001,
        "learning_rate_decay_epoch": 1,
        "learning_rate_decay_rate": 0.7,
        "max_to_keep": 3,
        "model_saving_interval": 1,
        "summary_step": 512,
        "printing_step": 512,
        "save_model_path": checkpoint_prefix

    }
    tfr_info = {
        "tfr_file_prefix": "part",
        "rows": None,
        "tfr_path": tfrecords_prefix,
        "columns": N_COLS[sensor_group_name],
        "interval": 1,
        "column_names": {"id_key": "store_name", "datetime_key": "date_time", "feature_key": "scaledFeatures",
                         "label_key": "label"}
    }

    main(train_param, tfr_info)
