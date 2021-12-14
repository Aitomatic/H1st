import datetime
import json
import os
import re
import sys
import time
import math
import glob
import boto3
import numpy as np
import pandas as pd
import s3fs

from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.decomposition import PCA
# import tensorflow as tf


# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf

tf.executing_eagerly()
# tf.disable_v2_behavior()

# tf.compat.v2.disable_resource_variables()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import input_utils

sys.path.append('../dl')

from nn import VAE

s3 = boto3.resource("s3")
s3_fs = s3fs.S3FileSystem()

pd.set_option('mode.chained_assignment', None)


def generate_ewma_resample1D_ad_score(df, id_col="equipment_instance_id", date_time_col="date_time"):
    gb = df.groupby(id_col)
    list_final = [gb.get_group(x) for x in gb.groups]
    df_list = []
    for idx, df_one in enumerate(list_final):
        df_one.index = df_one['date_time']
        df_one = df_one.drop('date_time', axis=1)
        temp = df_one.resample('D').mean()
        temp = ewma_10d(temp, avg_days=8)
        #         temp = temp.drop('ewma10__equipment_instance_id__VAE', axis = 1)
        temp[id_col] = df_one[id_col].iloc[0]
        df_list.append(temp)
    return pd.concat(df_list, axis=0)


def ewma_10d(tmp_df, avg_days=8):
    for col in tmp_df.columns.values:
        ewma = pd.Series.ewm
        fwd = ewma(tmp_df.loc[:, col], span=avg_days).mean()  # take EWMA in fwd direction
        bwd = ewma(tmp_df.loc[:, col][::-1], span=avg_days).mean()  # take EWMA in bwd direction
        filtered = np.vstack((fwd.values, bwd[::-1].values))  # lump fwd and bwd together
        filtered = np.mean(filtered, axis=0)  # average
        tmp_df['ewma8__dailyMean__' + col + '__lpg_water_system__old_filter_power_gen_on__VAE'] = filtered
    return tmp_df


class PredictiveMaintenanceVAE(object):
    def __init__(self, input_dim, n_window=1, h_dim=None, z_dim=None, batch_norm=True,
                 n_layer=3, alpha=0.00001, n_rank=0, pos_enc=False, drop_rate=0.0):

        self.pos_enc = pos_enc
        self.z_dim = z_dim
        if not z_dim:
            self.z_dim = self.get_proper_z_dim(input_dim=input_dim * n_window)
            # train_param['z_dim'] = z_dim
        self.h_dim = h_dim
        if not h_dim:
            self.h_dim = self.get_proper_hidden_dim(input_dim=input_dim * n_window)
            # train_param['h_dim'] = h_dim
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            # with tf.variable_scope("vae"):
            self.vae = VAE(
                encoding_networks=n_layer * [self.h_dim],
                decoding_networks=n_layer * [self.h_dim],
                n_input=input_dim * n_window,
                n_z=self.z_dim,
                bn=batch_norm,
                alpha=alpha,
                drop_rate=drop_rate,
                n_rank=n_rank)

        if self.pos_enc:
            self.position_encoding = self.get_position_encoding(
                num_positions=n_window,
                num_features=input_dim,
                min_val=10000)

        # self.position_encoding = self.get_position_encoding(
        #     num_positions=n_window,
        #     num_features=input_dim,
        #     min_val=10000)        
        self.model_param = {'n_input': input_dim,
                            'n_window': n_window,
                            'n_z': self.z_dim,
                            'h_dim': self.h_dim,
                            'n_layer': n_layer,
                            'batch_norm': batch_norm,
                            'alpha': alpha,
                            'drop_rate': drop_rate,
                            'n_rank': n_rank,
                            'pos_enc': self.pos_enc}

    def train(self, input_path, save_path, tfr_id_key, tfr_datetime_key, tfr_feature_key, pre_fix,
              data_cache, n_feature, n_window, n_row, batch_size, interval, num_epochs, beta,
              learning_rate_init, learning_rate_decay_epoch, learning_rate_decay_rate, max_to_keep,
              model_saving_interval, summary_step, printing_step):
        self.save_json(
            json_object=self.model_param,
            folder_path=save_path,
            json_name='model_param.json')
        logs_path = save_path
        val_loss_list = []

        with self.tf_graph.as_default():
            x_train = tf.keras.Input(name="x_placeholder", shape=[n_feature * n_window], dtype=tf.dtypes.float32)
            x_val = tf.keras.Input(name="x_placeholder_val", shape=[n_feature * n_window], dtype=tf.dtypes.float32)
            train_iter, val_iter, batchs_train, batchs_val = \
                self.get_train_val_iterator_from_tfr(
                    input_path, batch_size, tfr_id_key, tfr_datetime_key, tfr_feature_key,
                    n_feature, n_window, pre_fix, n_row, data_cache)
            x_train_tensor = train_iter.get_next()

            x_val_tensor = val_iter.get_next()

            global_step = tf.compat.v1.train.create_global_step()
            learning_rate = tf.compat.v1.train.exponential_decay(
                learning_rate_init,
                global_step,
                decay_steps=batchs_train * learning_rate_decay_epoch,
                decay_rate=learning_rate_decay_rate,
                staircase=True)

            with tf.compat.v1.variable_scope("vae"):
                train_loss, train_mae, train_mae_update_op = self.vae.get_mc_loss_n_mae(
                    tf.cast(x_train, tf.float32), beta=beta, training=True)
                val_loss, val_mae, val_mae_update_op = self.vae.get_mc_loss_n_mae(
                    tf.cast(x_val, tf.float32), beta=beta, training=False)

            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(train_loss, global_step=global_step)

            running_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="vae/my_mae")
            print("running_vars:", running_vars)
            running_vars_initializer = tf.compat.v1.variables_initializer(var_list=running_vars)

            # Create a saver.
            saver = tf.compat.v1.train.Saver(max_to_keep=max_to_keep)
            init_g = tf.compat.v1.global_variables_initializer()

            tf.summary.scalar("train_loss", train_loss)
            tf.summary.scalar("train_mae", train_mae)
            tf.summary.scalar("val_loss", val_loss)
            tf.summary.scalar("val_mae", val_mae)
            tf.summary.scalar("learning_rate", learning_rate)
            merged_summary_op = tf.compat.v1.summary.merge_all()

            with tf.compat.v1.Session() as sess:
                sess.run(init_g)
                sess.run(running_vars_initializer)
                summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())

                print(">>> batchs_train_per_epoch = ", batchs_train)
                for epoch in range(num_epochs):
                    for i in range(batchs_train):
                        start_time = time.time()
                        # print("before run", tfr_feature_key)
                        x_train_feature = sess.run(x_train_tensor[tfr_feature_key])
                        # print(x_train_feature.shape)
                        # print("after run")
                        _, _, global_step_value = sess.run([train_op,
                                                            train_mae_update_op,
                                                            # val_mae_update_op,
                                                            global_step],
                                                           feed_dict={x_train: x_train_feature})
                        duration = time.time() - start_time
                        if i % summary_step == 0:
                            x_val_feature = sess.run(x_val_tensor[tfr_feature_key])
                            if n_window > 1:
                                x_val_datetime = sess.run(x_val_tensor[tfr_datetime_key])
                                x_val_feature = x_val_feature.reshape([-1, n_feature * n_window])

                                x_val_datetime = x_val_datetime.reshape([-1, n_window])
                                x_val_feature, _ = self.get_valid_window_batch(
                                    x_val_feature, x_val_datetime, n_window, 10 * batch_size, interval)
                                if self.pos_enc:
                                    x_val_feature += self.position_encoding
                                if x_val_feature.shape[0] == 0:
                                    continue
                            # print "x_val_feature.shape:", x_val_feature.shape
                            _ = sess.run(val_mae_update_op, feed_dict={x_val: x_val_feature})
                            loss_value, mae_value, loss_value_val, mae_value_val = sess.run(
                                [train_loss, train_mae, val_loss, val_mae],
                                feed_dict={x_train: x_train_feature, x_val: x_val_feature})
                            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                            #                             summary_writer.add_summary(summary, global_step_value)
                            val_loss_list.append(loss_value_val)

                        if i % printing_step == 0:
                            num_examples_per_step = batch_size
                            examples_per_sec = num_examples_per_step / duration
                            sec_per_batch = duration
                            format_str = (
                                '%s epoch: %d batch: %d, mae: %.5f, mae_val: %.5f, loss: %.5f, loss_val: %.5f (%.1f examples/sec; %.3f ''sec/batch)')
                            print(format_str % (
                                datetime.datetime.now(), epoch, i, mae_value, mae_value_val, loss_value, loss_value_val,
                                examples_per_sec, sec_per_batch))
                            sess.run(running_vars_initializer)

                    # Save the model checkpoint periodically.
                    if epoch % model_saving_interval == 0 or (epoch + 1) == num_epochs:
                        checkpoint_path = os.path.join(save_path, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=global_step_value)

        return val_loss_list

    def generate_summary_bak(self, df, id_col, date_time_col, prob_col, sensor_cols, threshold, top_n=10):
        gb = df.groupby(id_col)
        list_final = [gb.get_group(x) for x in gb.groups]
        df_list = []
        for idx, df_one in enumerate(list_final):
            temp = df_one.set_index(date_time_col).resample('1h').mean()
            temp.reset_index(inplace=True)
            temp.reset_index(drop=True, inplace=True)
            temp2 = temp[temp.prob < threshold]
            temp2['time_diff'] = temp2[date_time_col] - temp2[date_time_col].shift(1)
            temp3 = temp2[temp2.time_diff == datetime.timedelta(hours=1)]
            if temp3.shape[0] > 0:
                temp4 = temp3.set_index(date_time_col).resample('1d').mean() \
                    .dropna().reset_index()
                temp4[id_col] = df_one[id_col].iloc[0]
                if sensor_cols:
                    df_list.append(temp4[[id_col, date_time_col, prob_col] + list(sensor_cols)])
                else:
                    df_list.append(temp4[[id_col, date_time_col, prob_col]])

        df_ad_result = pd.concat(df_list, axis=0).reset_index(drop=True)
        if sensor_cols:
            argsort_list = []
            for row in df_ad_result[sensor_cols].values:
                argsort_list.append([sensor_cols[idx] for idx in row.argsort()[:top_n]])
            df_top_sensors = pd.DataFrame(
                argsort_list, columns=['top' + str(i + 1) for i in xrange(top_n)])
            df_summary = pd.concat(
                [df_ad_result[[id_col, date_time_col, prob_col]], df_top_sensors],
                axis=1)
        else:
            df_summary = df_ad_result[[id_col, date_time_col, prob_col]]
        return df_summary

    def generate_summary(self, df, id_col, date_time_col, prob_col, sensor_cols, threshold, top_n=10):
        gb = df.groupby(id_col)
        list_final = [gb.get_group(x) for x in gb.groups]

        def get_individual_summary(df_one):
            temp = df_one.set_index(date_time_col).resample('1h').mean()
            temp.reset_index(inplace=True)
            temp.reset_index(drop=True, inplace=True)
            temp2 = temp[temp.prob < threshold]
            temp2['time_diff'] = temp2[date_time_col] - temp2[date_time_col].shift(1)
            temp3 = temp2[temp2.time_diff == datetime.timedelta(hours=1)]
            if temp3.shape[0] > 0:
                temp4 = temp3.set_index(date_time_col).resample('1d').mean() \
                    .dropna().reset_index()
                temp4[id_col] = df_one[id_col].iloc[0]
                if sensor_cols:
                    return temp4[[id_col, date_time_col, prob_col] + list(sensor_cols)]
                else:
                    return temp4[[id_col, date_time_col, prob_col]]
            else:
                return None

        pool = Pool().imap
        individual_summary_results = pool(get_individual_summary, list_final)
        df_list = []
        for result in individual_summary_results:
            if result is not None:
                df_list.append(result)

        df_ad_result = pd.concat(df_list, axis=0).reset_index(drop=True)
        if sensor_cols:
            argsort_list = []
            for row in df_ad_result[sensor_cols].values:
                argsort_list.append([sensor_cols[idx] for idx in row.argsort()[:top_n]])
            df_top_sensors = pd.DataFrame(
                argsort_list, columns=['top' + str(i + 1) for i in xrange(top_n)])
            df_summary = pd.concat(
                [df_ad_result[[id_col, date_time_col, prob_col]], df_top_sensors],
                axis=1)
        else:
            df_summary = df_ad_result[[id_col, date_time_col, prob_col]]
        return df_summary

    def generate_ad_score(self, df, id_col, date_time_col, prob_col):
        gb = df.groupby(id_col)
        list_final = [gb.get_group(x) for x in gb.groups]
        df_list = []
        for idx, df_one in enumerate(list_final):
            df_one = df_one[[id_col, date_time_col, prob_col]]
            threshold_keep = df_one.prob.quantile([0.30]).values[0]
            df_one = df_one[df_one.prob < threshold_keep]
            df_one.loc[:, 'prob'] = df_one.prob.clip(lower=-500).values
            temp = df_one.set_index(date_time_col).resample('1d').mean()
            temp.reset_index(inplace=True)
            temp.reset_index(drop=True, inplace=True)
            temp[id_col] = df_one[id_col].iloc[0]
            df_list.append(temp)
        df_ad_score = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df_ad_score

    def get_threshold(self, df, id_col, date_time_col, resample='1h', portion=0.001):
        gb = df.groupby(id_col)
        list_final = [gb.get_group(x) for x in gb.groups]
        avg_prob_list = []
        for df_one in list_final:
            avg_prob = df_one.set_index(date_time_col).resample(resample).mean().prob.values
            avg_prob_list.append(avg_prob)
        avg_prob_final = np.concatenate(avg_prob_list)
        avg_prob_final = avg_prob_final[~np.isnan(avg_prob_final)]
        return np.quantile(avg_prob_final, portion, axis=0)

    def get_range_group(self, df, datetime_col):
        idx_list = list(df.index)
        start_idx = end_idx = idx_list[0]
        group_list = []
        for idx in idx_list[1:]:
            if (df.loc[idx, datetime_col] - df.loc[end_idx, datetime_col] == datetime.timedelta(days=1)) or \
                    (df.loc[idx, datetime_col] - df.loc[end_idx, datetime_col] == datetime.timedelta(days=2)):
                end_idx = idx
            else:
                group_list.append([start_idx, end_idx])
                start_idx = end_idx = idx
        group_list.append([start_idx, end_idx])
        return group_list

    def get_range_summary_for_one_event(self, df, id_col, datetime_col, sensor_cols, prob_col, top_n=10):
        score_dict = {}
        for idx, col in enumerate(df.iloc[:, 3:]):
            multi = len(sensor_cols) - idx
            for sensor, cnt in df[col].value_counts().to_dict().iteritems():
                if sensor in score_dict:
                    score_dict[sensor] += multi * cnt
                else:
                    score_dict[sensor] = multi * cnt

        sensors = []
        for key, value in reversed(sorted(score_dict.items(), key=lambda kv: (kv[1], kv[0]))):
            sensors.append(key)
        #     print "%s: %s" % (key, value)
        summary = [df[id_col].iloc[0],
                   df[datetime_col].iloc[0],
                   df[datetime_col].iloc[-1],
                   df[prob_col].mean()] \
                  + sensors[:top_n]
        return summary

    def get_range_summary_for_one_instance(self, df, group_list, id_col, datetime_col, sensor_cols, prob_col, top_n=10):
        summary_list = []
        for idxs in group_list:
            df_group = df.loc[idxs[0]:idxs[1]]
            summary = self.get_range_summary_for_one_event(
                df_group, id_col, datetime_col, sensor_cols, prob_col, top_n)
            summary_list.append(summary)
        df_range_summary = pd.DataFrame(
            summary_list,
            columns=[id_col, 'from_date', 'to_date', 'score'] + \
                    ['top' + str(i) for i in range(top_n)])
        return df_range_summary

    def get_range_summary_for_one_instance_only_prob(self, df, group_list, id_col, datetime_col, prob_col):
        summary_list = []
        for idxs in group_list:
            df_group = df.loc[idxs[0]:idxs[1]]
            summary = [df_group[id_col].iloc[0],
                       df_group[datetime_col].iloc[0],
                       df_group[datetime_col].iloc[-1],
                       df_group[prob_col].mean()]
            summary_list.append(summary)
        df_range_summary = pd.DataFrame(
            summary_list,
            columns=[id_col, 'from_date', 'to_date', 'score'])
        return df_range_summary

    def get_range_summary_mlt(self, df, id_col, datetime_col, sensor_cols, prob_col, top_n=10):
        gb = df.groupby(id_col)
        list_final = [gb.get_group(x) for x in gb.groups]

        def get_individual_range_summary(df_one):
            df_one[datetime_col] = pd.to_datetime(df_one[datetime_col], utc=False)
            df_one.sort_values(datetime_col, inplace=True)
            df_one.reset_index(drop=True, inplace=True)
            list_group = self.get_range_group(df_one, datetime_col)
            if sensor_cols:
                df_range_summary = self.get_range_summary_for_one_instance(
                    df_one, list_group, id_col, datetime_col, sensor_cols, prob_col, top_n=top_n)
            else:
                df_range_summary = self.get_range_summary_for_one_instance_only_prob(
                    df_one, list_group, id_col, datetime_col, prob_col)
            return df_range_summary

        # return get_individual_range_summary(list_final[0])
        pool = Pool().imap
        individual_range_summary_results = pool(get_individual_range_summary, list_final)
        list_range_summary_df = []
        for result in individual_range_summary_results:
            list_range_summary_df.append(result)

        return pd.concat(list_range_summary_df, axis=0)

    def get_range_summary(self, df, id_col, datetime_col, sensor_cols, prob_col, top_n=10):
        gb = df.groupby(id_col)
        list_final = [gb.get_group(x) for x in gb.groups]
        list_range_summary_df = []
        for df_one in list_final:
            df_one[datetime_col] = pd.to_datetime(df_one[datetime_col], utc=False)
            df_one.sort_values(datetime_col, inplace=True)
            df_one.reset_index(drop=True, inplace=True)
            list_group = self.get_range_group(df_one, datetime_col)
            if sensor_cols:
                df_range_summary = self.get_range_summary_for_one_instance(
                    df_one, list_group, id_col, datetime_col, sensor_cols, prob_col, top_n=top_n)
            else:
                df_range_summary = self.get_range_summary_for_one_instance_only_prob(
                    df_one, list_group, id_col, datetime_col, prob_col)
            list_range_summary_df.append(df_range_summary)
        return pd.concat(list_range_summary_df, axis=0)

    def score(self, selected_columns, input_path, model_path, save_path, tfr_id_key, tfr_datetime_key,
              tfr_feature_key, n_feature, n_window, interval, n_row, tfr_score_index, pre_fix, checkpoint=None,
              num_sample=20,
              batch_size=2048, individual_sensor_prob=False, summary=False):
        if (not os.path.exists(save_path)) and ("s3://" not in save_path):
            os.mkdir(save_path)

        with self.tf_graph.as_default():
            x = tf.keras.Input(name="x_placeholder", shape=[n_feature * n_window], dtype=tf.dtypes.float32)
            iterator, batchs_per_epoch = self.get_iterator_from_tfr(
                input_path, batch_size, tfr_id_key, tfr_datetime_key,
                tfr_feature_key, n_feature, n_window, n_row, tfr_score_index, pre_fix)
            x_tensor = iterator.get_next()
            x_log_prob = self.vae(x, num_sample=num_sample, training=False)
            z_mean, _ = self.vae.get_z_param(x, training=False)
            prob_for_individual_sensor = self.vae.get_prob_for_individual_sensor(
                x, num_sensors=n_feature, num_sample=num_sample, window_size=n_window, training=False)

            saver = tf.compat.v1.train.Saver()
            init_g = tf.compat.v1.global_variables_initializer()

            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(init_g)

                # Model restore
                checkpoint_dir = model_path
                if not checkpoint:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
                    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                else:
                    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint)

                saver.restore(sess, latest_checkpoint)

                # Prediction
                id_list = []
                datetime_list = []
                log_prob_avg_list = []
                z_mean_avg_list = []
                sensor_avg_list = []
                print("batchs_per_epoch: ", batchs_per_epoch)
                for i in range(batchs_per_epoch):
                    if i % int(batchs_per_epoch / 20) == 0:
                        print("{}: {}% completed".format(datetime.datetime.now(),
                                                         int(np.ceil(100 * (i) / float(batchs_per_epoch)))))
                    x_input = sess.run(x_tensor)
                    x_score_feature = x_input[tfr_feature_key]

                    if individual_sensor_prob:
                        x_log_prob_o, prob_for_individual_sensor_o, z_mean_o = \
                            sess.run([x_log_prob, prob_for_individual_sensor, z_mean],
                                     feed_dict={x: x_score_feature})
                        sensor_avg_list.extend(prob_for_individual_sensor_o)
                    else:
                        x_log_prob_o, z_mean_o = \
                            sess.run([x_log_prob, z_mean],
                                     feed_dict={x: x_score_feature})
                    if n_window > 1:
                        id_list.extend(x_score_id[x_score_idx][:, -1])
                        datetime_list.extend(x_score_datetime[x_score_idx][:, -1])
                    else:
                        id_list.extend(x_input[tfr_id_key])
                        datetime_list.extend(x_input[tfr_datetime_key])
                    log_prob_avg_list.extend(x_log_prob_o)
                    z_mean_avg_list.extend(z_mean_o)

        print("length of id_list:", len(id_list))
        print("length of datetime_list:", len(datetime_list))
        print("length of log_prob_avg_list:", len(log_prob_avg_list))
        print("length of z_mean_avg_list:", len(z_mean_avg_list))
        if individual_sensor_prob:
            print("length of sensor_avg_list:", len(sensor_avg_list))
            print("length of sensor_avg_list[0]:", len(sensor_avg_list[0]))

        z_mean_avg_array = np.array(z_mean_avg_list)
        z_mean_avg_list = None
        if len(z_mean_avg_array[0]) > 2:
            pca = PCA(n_components=2)
            pca.fit(z_mean_avg_array)
            z_pca = pca.transform(z_mean_avg_array)
        elif len(z_mean_avg_array[0]) == 2:
            z_pca = z_mean_avg_array
        elif len(z_mean_avg_array[0]) == 1:
            z_pca = np.squeeze(np.transpose(np.array([z_mean_avg_array, z_mean_avg_array])))
        else:
            print("z_dim is zero or negative")

        prob = np.array(log_prob_avg_list).reshape((-1, 1))
        log_prob_avg_list = None
        if individual_sensor_prob:
            sensor_prob = np.array(sensor_avg_list).reshape((-1, n_feature))
            sensor_cols = selected_columns
            #             sensor_cols = ['sensor_'+ str(i) for i in range(len(sensor_avg_list[0]))]
            sensor_avg_list = None
            result_np = np.concatenate((prob, z_pca, sensor_prob), axis=1)
            result_df = pd.DataFrame(data=result_np, columns=['prob'] + ['z1', 'z2'] + sensor_cols)
        else:
            result_np = np.concatenate((prob, z_pca), axis=1)
            result_df = pd.DataFrame(data=result_np, columns=['prob'] + ['z1', 'z2'])
            sensor_cols = None
        result_np = None
        df_id_n_datetime = pd.DataFrame({tfr_id_key: id_list,
                                         'date_time': datetime_list})
        id_list = None
        datetime_list = None
        df_whole = pd.concat([df_id_n_datetime, result_df], axis=1)

        df_whole['date_time'] = pd.to_datetime(df_whole['date_time'].str.decode('UTF-8'))
        df_whole[tfr_id_key] = df_whole[tfr_id_key].str.decode('UTF-8')
        print("df_whole.shape:", df_whole.shape)
        output_path = save_path + '/VAEAnomScores30Minutes.parquet'
        print(output_path)
        df_whole.to_parquet(output_path)

    def get_position_encoding(self, num_positions, num_features, min_val=10000):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(min_val, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(0, num_positions)[:, np.newaxis],
                                np.arange(0, num_features)[np.newaxis, :],
                                num_features)
        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding.reshape([1, -1])
        # pos_encoding = pos_encoding[np.newaxis, :]    
        # return pos_encoding
        return pos_encoding / 5
        # return tf.cast(pos_encoding, dtype=tf.float32)

    def get_proper_z_dim(self, input_dim):
        return int(max(1, math.ceil(input_dim / 8.0)))

    def get_proper_hidden_dim(self, input_dim):
        return int(min(512, pow(2, math.ceil(math.log(input_dim, 2)) + 2)))

    def _count_record(self, files):
        c = 0
        #         with tf.compat.v1.Session() as sess:
        for i, fn in enumerate(files):
            if i % 100 == 0:
                print(" File : ", i)
            for record in tf.compat.v1.python_io.tf_record_iterator(fn):
                c += 1
        return c

    def _decode(self, serialized_example, decode_key, n_feature=None):

        features = tf.io.parse_single_example(
            serialized_example,
            features={decode_key: tf.io.FixedLenFeature([n_feature], dtype=tf.float32)})
        return features

    def get_train_val_iterator_from_tfr_in_s3_new(self, folder_path, batch_size,
                                                  tfr_id_key, tfr_datetime_key, tfr_feature_key, n_feature, n_window,
                                                  pre_fix, n_row=None, cache=False):
        if pre_fix[-1] != '*':
            pre_fix += '*'
        filenames = ["s3://" + k for k in s3_fs.glob(os.path.join(folder_path, pre_fix))]
        # print "filenames:", filenames
        if not filenames:
            print("Couldn't get any data from the provided path:", os.path.join(folder_path, pre_fix))
            exit()
        if not n_row:
            DATASET_SIZE = self._count_record(filenames)
        else:
            DATASET_SIZE = n_row
        print("DATASET_SIZE:", DATASET_SIZE)
        num_parallel_reads = 8 if n_window == 1 else 1
        full_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_reads)

        full_dataset = full_dataset.shuffle(100 * batch_size + 1000)
        _tmp_full_dataset = lambda x: self._decode(x, decode_key=tfr_feature_key, n_feature=n_feature)

        full_dataset = full_dataset.map(_tmp_full_dataset, num_parallel_calls=32)

        train_size = int(0.99 * DATASET_SIZE)
        val_size = int(0.01 * DATASET_SIZE)
        print("train_size: ", train_size, "val_size: ", val_size)
        batchs_per_train = int(train_size / (batch_size * n_window)) + 1
        batchs_per_val = int(val_size / (batch_size * n_window)) + 1
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)

        if cache:
            train_dataset = train_dataset.cache()
            val_dataset = val_dataset.cache()

        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(batch_size * n_window)
        train_dataset = train_dataset.prefetch(4)
        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)

        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(10 * batch_size * n_window)
        val_dataset = val_dataset.prefetch(4)
        val_iterator = tf.compat.v1.data.make_one_shot_iterator(val_dataset)

        return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]

    def get_train_val_iterator_from_tfr_in_local(self, folder_path, batch_size,
                                                 tfr_id_key, tfr_datetime_key, tfr_feature_key, n_feature, n_window,
                                                 pre_fix, n_row=None, cache=False):
        if pre_fix[-1] != '*':
            pre_fix += '*'
        filenames = glob.glob(os.path.join(folder_path, pre_fix))

        if not filenames:
            print("Couldn't get any data from the provided path.")
            exit()
        if not n_row:
            DATASET_SIZE = self._count_record(filenames)
        else:
            DATASET_SIZE = n_row
        print("DATASET_SIZE:", DATASET_SIZE)
        num_parallel_reads = 8 if n_window == 1 else 1
        full_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_reads)

        full_dataset = full_dataset.shuffle(100 * batch_size + 1000)
        _tmp_full_dataset = lambda x: self._decode(x, decode_key=tfr_feature_key, n_feature=n_feature)

        full_dataset = full_dataset.map(_tmp_full_dataset, num_parallel_calls=32)

        train_size = int(0.99 * DATASET_SIZE)
        val_size = int(0.01 * DATASET_SIZE)
        print("train_size: ", train_size, "val_size: ", val_size)
        batchs_per_train = int(train_size / (batch_size * n_window)) + 1
        batchs_per_val = int(val_size / (batch_size * n_window)) + 1
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)

        if cache:
            train_dataset = train_dataset.cache()
            val_dataset = val_dataset.cache()

        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(batch_size * n_window)
        train_dataset = train_dataset.prefetch(4)
        #         train_iterator = iter(train_dataset)
        train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)

        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(10 * batch_size * n_window)
        val_dataset = val_dataset.prefetch(4)
        #         train_iterator = iter(val_dataset)
        val_iterator = tf.compat.v1.data.make_one_shot_iterator(val_dataset)

        return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]

    # To Do: n_window version is not implemented
    def get_train_val_iterator_from_tfr_in_s3(self, folder_path, batch_size,
                                              decode_key, n_feature, n_window, pre_fix, n_row, cache=False):
        print("folder_path:", folder_path)
        result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
        bucket_name = result.group(1)
        tfr_path = result.group(2)

        client = boto3.client('s3')
        response = client.list_objects(Bucket=bucket_name,
                                       Prefix=os.path.join(tfr_path, pre_fix))
        # print "response:", response
        filenames = [os.path.join('s3://', bucket_name, item['Key']) for item in response['Contents']]
        # print "filenames length:", len(filenames)
        # print filenames[:2]

        DATASET_SIZE = round(n_row)
        full_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=8)
        full_dataset = full_dataset.shuffle(100 * batch_size + 1000)
        _tmp_full_dataset = lambda x: self._decode2(x, decode_key=decode_key, n_feature=n_feature)
        full_dataset = full_dataset.map(_tmp_full_dataset, num_parallel_calls=32)
        train_size = int(0.99 * DATASET_SIZE)
        val_size = int(0.01 * DATASET_SIZE)
        print("train_size: ", train_size, "val_size: ", val_size)
        batchs_per_train = int(train_size / batch_size) + 1
        batchs_per_val = int(val_size / batch_size) + 1
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)

        if cache:
            train_dataset = train_dataset.cache()
            val_dataset = val_dataset.cache()

        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(4)
        train_iterator = train_dataset.make_one_shot_iterator()

        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(10 * batch_size)
        val_dataset = val_dataset.prefetch(4)
        val_iterator = val_dataset.make_one_shot_iterator()

        return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]

    def get_train_val_iterator_from_tfr(self, folder_path, batch_size,
                                        tfr_id_key, tfr_datetime_key, tfr_feature_key, n_feature,
                                        n_window, pre_fix, n_row, cache=False):
        if "s3://" in folder_path:
            train_iter, val_iter, batchs_train, batchs_val = \
                self.get_train_val_iterator_from_tfr_in_s3_new(
                    folder_path, batch_size, tfr_id_key, tfr_datetime_key, tfr_feature_key, n_feature,
                    n_window, pre_fix, n_row, cache)
        else:
            train_iter, val_iter, batchs_train, batchs_val = \
                self.get_train_val_iterator_from_tfr_in_local(
                    folder_path, batch_size, tfr_id_key, tfr_datetime_key,
                    tfr_feature_key, n_feature, n_window, pre_fix, n_row, cache)

        return [train_iter, val_iter, batchs_train, batchs_val]

    def save_json(self, json_object, folder_path, json_name):
        if 's3://' in folder_path:
            result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
            bucket_name = result.group(1)
            file_path = result.group(2)
            obj = s3.Object(bucket_name, os.path.join(file_path, json_name))
            obj.put(Body=json.dumps(json_object))
        else:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            with open(os.path.join(folder_path, json_name), 'w') as outfile:
                json.dump(json_object, outfile)

    def _decode2(self, serialized_example, id_key, datetime_key, feature_key, n_input=None):
        unserialized = tf.io.parse_single_example(
            serialized_example,
            features={id_key: tf.io.FixedLenFeature([], dtype=tf.string),
                      datetime_key: tf.io.FixedLenFeature([], dtype=tf.string),
                      feature_key: tf.io.FixedLenFeature([n_input], dtype=tf.float32)
                      })
        return unserialized

    def get_iterator_from_tfr_in_s3(self, folder_path, batch_size, tfr_id_key,
                                    tfr_datetime_key, tfr_feature_key, n_input, n_window, n_row, pre_fix,
                                    is_train=False, cache=False):

        print("folder_path:", folder_path)
        result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
        bucket_name = result.group(1)
        tfr_path = result.group(2)
        client = boto3.client('s3')
        response = client.list_objects(Bucket=bucket_name,
                                       Prefix=os.path.join(tfr_path, pre_fix))
        filenames = [os.path.join('s3://', bucket_name, item['Key']) for item in response['Contents']]
        # print "filenames length:", len(filenames)
        # print filenames[:2]
        if not filenames:
            print("Couldn't get any data from the provided path.")
            exit()

        if not n_row:
            DATASET_SIZE = self._count_record(filenames)
        else:
            DATASET_SIZE = n_row
        batchs_per_train = int(DATASET_SIZE / batch_size) + 1
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
        _tmp_dataset = lambda x: self._decode2(
            x,
            tfr_id_key,
            tfr_datetime_key,
            tfr_feature_key,
            n_input=n_input)
        dataset = dataset.map(_tmp_dataset, num_parallel_calls=8)
        if is_train:
            dataset = dataset.shuffle(1000 + 3 * batch_size)
            if cache:
                dataset = dataset.cache()
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(4)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return [iterator, batchs_per_train]

    def get_iterator_from_tfr_in_local(self, folder_path, batch_size, tfr_id_key, tfr_datetime_key,
                                       tfr_feature_key, n_input, n_window, n_row, tfr_score_index, pre_fix,
                                       is_train=False, cache=False):
        if pre_fix[-1] != '*':
            pre_fix += '*'
        filenames = sorted(glob.glob(os.path.join(folder_path, pre_fix)))[tfr_score_index[0]:tfr_score_index[1]]
        print("Length", len(filenames))
        print("filenames:", filenames)
        if not filenames:
            print("Couldn't get any data from the provided path.")
            exit()
        if not n_row:
            DATASET_SIZE = self._count_record(filenames)
        else:
            DATASET_SIZE = n_row
        print("DATASET_SIZE:", DATASET_SIZE)
        batchs_per_epoch = int(DATASET_SIZE / (batch_size * n_window)) + 1
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
        _tmp_dataset = lambda x: self._decode2(
            x,
            tfr_id_key,
            tfr_datetime_key,
            tfr_feature_key,
            n_input=n_input)
        dataset = dataset.map(_tmp_dataset, num_parallel_calls=8)
        if is_train:
            dataset = dataset.shuffle(1000 + 3 * batch_size)
            if cache:
                dataset = dataset.cache()
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size * n_window)
        dataset = dataset.prefetch(4)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator, batchs_per_epoch

    def get_iterator_from_tfr(self, input_path, batch_size, tfr_id_key, tfr_datetime_key,
                              tfr_feature_key, n_feature, n_window, n_row, tfr_score_index, pre_fix, is_train=False,
                              cache=False):

        if "s3://" in input_path:
            iterator, batchs_per_epoch = self.get_iterator_from_tfr_in_s3(
                input_path, batch_size, tfr_id_key, tfr_datetime_key,
                tfr_feature_key, n_feature, n_window, n_row, pre_fix, is_train, cache)
        else:
            iterator, batchs_per_epoch = self.get_iterator_from_tfr_in_local(
                input_path, batch_size, tfr_id_key, tfr_datetime_key,
                tfr_feature_key, n_feature, n_window, n_row, tfr_score_index, pre_fix, is_train, cache)

        return [iterator, batchs_per_epoch]

    def save_df_to_csv_in_s3(self, dataframe, save_path, csv_name):
        result = re.search("s3:\/\/(.+?)\/(.*)", save_path)
        bucket_name = result.group(1)
        file_path = result.group(2)
        csv_buffer = BytesIO()
        dataframe.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket_name, os.path.join(file_path, csv_name)) \
            .put(Body=csv_buffer.getvalue())

    def get_valid_window_batch(self, x_feature, x_datetime, n_window, batch_size, interval):
        valid_window_idx = []
        for i in range(batch_size):
            datetime_start = datetime.datetime.strptime(x_datetime[i][0], "%Y-%m-%d %H:%M:%S").replace(second=0)
            datetime_end = datetime.datetime.strptime(x_datetime[i][-1], "%Y-%m-%d %H:%M:%S").replace(second=0)
            datetime_diff = datetime_end - datetime_start
            if datetime_diff == datetime.timedelta(minutes=interval * (n_window - 1)):
                valid_window_idx.append(i)
        return x_feature[valid_window_idx], valid_window_idx
