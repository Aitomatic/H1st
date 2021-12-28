from datetime import datetime, timedelta

import s3fs
import json
import os
import sys

from vae import PredictiveMaintenanceVAE

s3_fs = s3fs.S3FileSystem()

GROUP_COLS = {
    'group_1': [
        'condensing_pressure',
        'evaporation_pressure',
        'return_gas_temp',
        # '30_minutes_accumulated_power_consumption',
        'showcase_temp1_f5',
        'showcase_temp1_f7',
        'showcase_temp1_f9'
    ],
    'group_2': [
        'condensing_pressure',
        'evaporation_pressure',
        'return_gas_temp',
        # '30_minutes_accumulated_power_consumption',
        'showcase_temp1_f5',
        'showcase_temp1_f7',
        'showcase_temp1_f9',
        'showcase_temp1_f11'
    ],
    'group_5': [
        'condensing_pressure',
        'evaporation_pressure',
        # '30_minutes_accumulated_power_consumption',
        'showcase_temp1_f5',
        'showcase_temp1_f7',
        'showcase_temp1_f9',
        'showcase_temp1_f17',
        'showcase_temp2_f18'
    ],
}


def main(score_param, model_param, tfr_info):
    pm_vae = PredictiveMaintenanceVAE(
        input_dim=tfr_info["columns"],
        n_window=model_param["n_window"],
        h_dim=model_param["h_dim"],
        z_dim=model_param["n_z"],
        n_layer=model_param["n_layer"],
        batch_norm=model_param["batch_norm"],
        alpha=model_param["alpha"],
        n_rank=model_param["n_rank"],
    )

    pm_vae.score(
        selected_columns=score_param["selected_columns"],
        input_path=tfr_info["tfr_path"],
        model_path=score_param["model_path"],
        save_path=score_param["result_path"],
        tfr_id_key=tfr_info["column_names"]["id_key"],
        tfr_datetime_key=tfr_info["column_names"]["datetime_key"],
        tfr_feature_key=tfr_info["column_names"]["feature_key"],
        n_feature=tfr_info["columns"],
        n_window=model_param["n_window"],
        interval=tfr_info["interval"],
        n_row=tfr_info["rows"],
        tfr_score_index=tfr_info["tfr_score_index"],
        pre_fix=tfr_info["tfr_file_prefix"],
        checkpoint=score_param["model_checkpoint"],
        num_sample=score_param["num_sample"],
        batch_size=score_param["batch_size"],
        individual_sensor_prob=score_param["individual_sensor_prob"],
        summary=score_param["summary"],
    )


def date_range(start_time, end_time):
    for n in range(int((end_time - start_time).days) + 1):
        yield (start_time + timedelta(n)).strftime("%Y-%m-%d")


def to_dt(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


TFRECORDS_PREFIX = os.environ["TFRECORDS_PREFIX"]
CHECKPOINT_PREFIX = os.environ.get("CHECKPOINT_PREFIX")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX")

N_COLS = {
    'group_1': 6,
    'group_2': 7,
    'group_5': 7,
}

if __name__ == "__main__":
    sensor_group_name = sys.argv[1]
    operation_mode = 'Cooling'
    tfrecords_prefix = "%s.all/%s.tfrecords/operation_mode=%s" % (TFRECORDS_PREFIX, sensor_group_name, operation_mode)
    checkpoints_prefix = "%s/%s/operation_mode=%s" % (CHECKPOINT_PREFIX, sensor_group_name, operation_mode)
    output_prefix = "%s.all/%s/operation_mode=%s" % (OUTPUT_PREFIX, sensor_group_name, operation_mode)

    if len(sys.argv) > 2:
        upload_date = sys.argv[2]
        tfrecords_prefix = "%s/%s.tfrecords/operation_mode=%s/upload_date=%s" % (
            TFRECORDS_PREFIX, sensor_group_name, operation_mode, upload_date)
        output_prefix = "%s/%s/operation_mode=%s/upload_date=%s" % (
            OUTPUT_PREFIX, sensor_group_name, operation_mode, upload_date)

    selected_columns = GROUP_COLS[sensor_group_name]
    print(selected_columns, len(selected_columns))

    for score_idx in range(0, 200, 200):
        score_param = {
            "model_path": checkpoints_prefix,
            "model_checkpoint": None,
            "result_path": output_prefix,
            "num_sample": 20,
            "batch_size": 64,
            "selected_columns": selected_columns,
            "individual_sensor_prob": True,
            "summary": True,
        }
        tfr_info = {
            "tfr_score_index": [score_idx, score_idx + 200],
            "tfr_file_prefix": "part",
            "rows": None,
            "tfr_path": tfrecords_prefix,
            "columns": N_COLS[sensor_group_name],
            "interval": 1,
            "column_names": {
                "id_key": "store_name",
                "datetime_key": "date_time",
                "feature_key": "scaledFeatures",
                "label_key": "label",
            },
        }

        model_param_json_path = os.path.join(
            score_param["model_path"], "model_param.json"
        )

        with s3_fs.open(model_param_json_path) as json_file:
            model_param = json.load(json_file)

        main(score_param, model_param, tfr_info)
