import numpy as np
import pandas as pd
import s3fs
import sys

s3_fs = s3fs.S3FileSystem()

STORES = {'dalian_lawson_kaifaqujumeidongwan': [1, 2],
          'dalian_lawson_dongruanruanjianyuan': [1, 2],
          'dalian_lawson_renminlubaiyujie': [1, 2],
          'dalian_lawson_qixianlingruifengyuan': [1, 2],
          'dalian_lawson_guangxianlusaiboledaxia': [1, 2],
          'dalian_lawson_huangpuluyinhaiwanxiang': [1, 5],
          'dalian_lawson_caishijiezhenfudaxia': [1, 5],
          'dalian_lawson_jiazhaoyeguangchang': [1, 2],
          'dalian_lawson_hongxinghailangu': [1, 2],
          'dalian_lawson_xiandaifuwuyedaxia': [1, 2],
          'dalian_lawson_kaifaqujiangchengguangchang': [1, 2],
          'dalian_lawson_jinzhouwankecheng': [1, 2],
          'dalian_lawson_qingnierjie': [1, 2],
          'dalian_lawson_rongshengjie': [1, 5],
          'dalian_lawson_tianjinjietianheguangchang': [1, 5],
          'dalian_lawson_taidedasha': [1, 2, 5],
          'dalian_lawson_kaifaqushuiyulanting': [1, 2],
          'dalian_lawson_xinghaibainianhui': [1, 5],
          'dalian_lawson_titanlunuodedaxia': [1, 2],
          'dalian_lawson_haikoulupenghui': [1, 2],
          'dalian_lawson_xiaopingdaobofeilandao': [1, 2],
          'dalian_lawson_huashunjiexiangzhouxincheng': [1, 5],
          'dalian_lawson_kaifaqupuxiangitzhongxin': [1, 2],
          'dalian_lawson_huarunkaixuanmenyiqi': [1, 5],
          'dalian_lawson_yuanyangfengjing': [1, 2],
          'dalian_lawson_jinzhouxinxiwangcheng': [1, 2],
          'dalian_lawson_shandongluhuananzhongxue': [1, 2],
          'dalian_lawson_renminluanlejie': [1, 2],
          'dalian_lawson_songjiangluhuazhongjie': [1, 2],
          'dalian_lawson_yidaeryuan': [1, 2],
          'dalian_lawson_xianlukejiguangchang': [1, 2],
          'dalian_lawson_xianluxingzhengdaxia': [1],
          'dalian_lawson_huanghelujiaotongdaxue': [1, 2],
          'dalian_lawson_xiaopingdaohaijun': [1, 5],
          'dalian_lawson_dunhuangluhuaxinyuan': [1, 2],
          'dalian_lawson_xinggongjie': [1, 2],
          'dalian_lawson_changpingjie': [1, 2],
          'dalian_lawson_donggangshuijingliwan': [1, 2],
          'dalian_lawson_haizhongguoxingyunjie': [1, 2],
          'dalian_lawson_shenyangsanhaojieqinghuatongfang': [1, 2],
          'dalian_lawson_hutanlubitaobeiyuan': [1, 2],
          'dalian_lawson_ganjingzitiyuzhongxin': [1, 5],
          'dalian_lawson_jinzhouwenrunjinchen': [1, 5],
          'dalian_lawson_ganjingziqufengdanlicheng': [1, 2],
          'dalian_lawson_jiefangluqingyunyingshan': [1, 2],
          'dalian_lawson_kaifaquwanda': [1],
          'dalian_lawson_fujingjiexingfuejia': [1, 2],
          'dalian_lawson_shenyangcaifuzhongxin': [1, 5],
          'dalian_lawson_shengliqiaobei': [1, 2],
          'dalian_lawson_jinzhouhepingliyuan': [1, 2],
          'dalian_lawson_kaifaquxinjie': [1, 2],
          'dalian_lawson_gaoxinqujingxianjie': [1, 5],
          'dalian_lawson_hongxinghaishiguanghai': [1, 2],
          'dalian_lawson_kaifaqubenxijie': [1, 2],
          'dalian_lawson_shenyangxita': [1, 2],
          'dalian_lawson_shenyangjiandayunfeng': [1, 2],
          'dalian_lawson_xiuyuejiemingxiushanzhuang': [1, 2],
          'dalian_lawson_shenyangyunfengbeijie': [1, 2],
          'dalian_lawson_huitongjieyuanyangrongyu': [1, 2],
          'dalian_lawson_gaoxinyuanquziyuguandi': [1, 2],
          'dalian_lawson_dashangqingnijie': [1, 2],
          'dalian_lawson_kaifaquxingchenglu': [1, 2],
          'dalian_lawson_tianheluhuadongrenjia': [1, 2]}


# Compute mean using 'weight' or 'harmonic' method
def compute_mean(list_array, weights, method='weight'):
    if len(list_array) == len(weights):
        if method == 'weight':
            weight_avg = np.sum((np.array(list_array)) * (np.array(weights).reshape(-1, 1)), axis=0) / np.sum(weights)
        elif method == 'harmonic':
            weight_avg = np.sum(weights) / np.sum(np.array(weights).reshape(-1, 1) / (np.array(list_array)), axis=0)
        else:
            print('There are only two methods to compute mean: "weight" and "harmonic".')
            return None
        return weight_avg

    print('Please check, length of list_array and length of weights must be equal, check weights!!')
    return None


# combine anomaly score from different groups by using 'weight' or 'harmonic' mean
def combine_anom_score(stores, ad_gr1, ad_gr2, ad_gr5, w12=None, w15=None, w125=None, method='weight'):
    """
    stores: {store_name: [group_x, group_y]}
    ad_gr1: anomaly score from model vae, having three features ['store_name', 'date_time', 'prob']
    w12: weights on the 'prob' when average
    method: 'weight' or 'harmonic'
    """
    if w125 is None:
        w125 = [1, 1, 1]
    if w15 is None:
        w15 = [1, 1]
    if w12 is None:
        w12 = [1, 1]

    # copy dataframe to avoid override their values
    ad_g1 = ad_gr1.copy()
    ad_g2 = ad_gr2.copy()
    ad_g5 = ad_gr5.copy()

    # offset to adjust probability to positive
    offset = np.ceil(abs(min([ad_g1.prob.min(), ad_g2.prob.min(), ad_g5.prob.min()])))
    ad_g1['prob'] = ad_g1[['prob']] + offset
    ad_g2['prob'] = ad_g2[['prob']] + offset
    ad_g5['prob'] = ad_g5[['prob']] + offset

    # start to combine probability
    df = pd.DataFrame(columns=['store_name', 'date_time', 'prob'])
    for shop_name, list_group in stores.items():
        if len(list_group) == 1:
            tmp_df = ad_g1[ad_g1['store_name'] == shop_name].copy()
            df = df.append(tmp_df[['store_name', 'date_time', 'prob']], ignore_index=True)
        elif len(list_group) == 2:
            if list_group == [1, 2]:
                tmp_df = ad_g1[ad_g1['store_name'] == shop_name].copy()
                if (len(ad_g1[ad_g1['store_name'] == shop_name].prob.values)
                        == len(ad_g2[ad_g2['store_name'] == shop_name].prob.values)):
                    weight_avg = compute_mean([ad_g1[ad_g1['store_name'] == shop_name].prob.values,
                                               ad_g2[ad_g2['store_name'] == shop_name].prob.values], w12, method)
                    if weight_avg is None:
                        return
                    tmp_df['prob'] = weight_avg
                else:
                    print('lack of data in anomaly group: ', shop_name, list_group)
                df = df.append(tmp_df[['store_name', 'date_time', 'prob']], ignore_index=True)
            elif list_group == [1, 5]:
                tmp_df = ad_g1[ad_g1['store_name'] == shop_name].copy()
                if (len(ad_g1[ad_g1['store_name'] == shop_name].prob.values)
                        == len(ad_g5[ad_g5['store_name'] == shop_name].prob.values)):
                    weight_avg = compute_mean([ad_g1[ad_g1['store_name'] == shop_name].prob.values,
                                               ad_g5[ad_g5['store_name'] == shop_name].prob.values], w15, method)
                    if weight_avg is None:
                        return
                    tmp_df['prob'] = weight_avg
                else:
                    print('lack of data in anomaly group: ', shop_name, list_group)
                df = df.append(tmp_df[['store_name', 'date_time', 'prob']], ignore_index=True)
            else:
                print('Please check the unique list groups! There is no: ', list_group)
        elif len(list_group) == 3:
            if list_group == [1, 2, 5]:
                tmp_df = ad_g1[ad_g1['store_name'] == shop_name].copy()
                if ((len(ad_g1[ad_g1['store_name'] == shop_name].prob.values)
                     == len(ad_g2[ad_g2['store_name'] == shop_name].prob.values)) and
                        (len(ad_g2[ad_g2['store_name'] == shop_name].prob.values)
                         == len(ad_g5[ad_g5['store_name'] == shop_name].prob.values))):
                    weight_avg = compute_mean([ad_g1[ad_g1['store_name'] == shop_name].prob.values,
                                               ad_g2[ad_g2['store_name'] == shop_name].prob.values,
                                               ad_g5[ad_g5['store_name'] == shop_name].prob.values], w125, method)
                    if weight_avg is None:
                        return
                    tmp_df['prob'] = weight_avg
                else:
                    print('lack of data in anomaly group: ', shop_name, list_group)
                df = df.append(tmp_df[['store_name', 'date_time', 'prob']], ignore_index=True)
            else:
                print('Please check the unique list groups! There is no: ', list_group)
        else:
            print('Please check the unique list groups! There is no: ', list_group)
    return df


def read_df_from_s3(s3_path):
    return pq.ParquetDataset(s3_path, filesystem=s3_fs).read_pandas().to_pandas()


INPUT_PREFIX = "s3://arimo-panasonic-ap-cn-cc-pm/.arimo/PredMaint/VAE/results"
OUTPUT_PREFIX = "s3://arimo-panasonic-ap-cn-cc-pm/.arimo/PredMaint/VAE/combined_results"


def main(operation_mode='Cooling', upload_date=None):
    input_prefix = "%s.all/__GROUP_NAME__/operation_mode=%s/VAEAnomScores30Minutes.parquet" % (
        INPUT_PREFIX, operation_mode)
    output_prefix = "%s.all/operation_mode=%s/VAEAnomScores30Minutes.parquet" % (OUTPUT_PREFIX, operation_mode)

    if upload_date:
        input_prefix = "%s/__GROUP_NAME__/operation_mode=%s/upload_date=%s/VAEAnomScores30Minutes.parquet" % (
            INPUT_PREFIX, operation_mode, upload_date)
        output_prefix = "%s/operation_mode=%s/upload_date=%s/VAEAnomScores30Minutes.parquet" % (
            OUTPUT_PREFIX, operation_mode, upload_date)

    # read anomaly score from 3 groups
    ad_gr1 = read_df_from_s3(input_prefix.replace('__GROUP_NAME__', 'group_1'))
    ad_gr2 = read_df_from_s3(input_prefix.replace('__GROUP_NAME__', 'group_2'))
    ad_gr5 = read_df_from_s3(input_prefix.replace('__GROUP_NAME__', 'group_5'))
    print('The number of stores in:')
    print('Group1: ', len(ad_gr1['store_name'].unique()))
    print('Group2: ', len(ad_gr2['store_name'].unique()))
    print('Group5: ', len(ad_gr5['store_name'].unique()))

    # run combined 3 groups: group1, group2, and group5
    combined_df = combine_anom_score(STORES, ad_gr1, ad_gr2, ad_gr5,
                                     w12=[0.4, 0.6], w15=[0.4, 0.6], w125=[0.2, 0.4, 0.4], method='harmonic')
    # print(combined_df)
    combined_df.to_parquet(output_prefix)
    return output_prefix


def lambda_handler(event, context):
    return {
        "error_code": 200,
        "results": main(upload_date=event.get('upload_date'))
    }


if __name__ == '__main__':
    print(main(upload_date=sys.argv[1] if len(sys.argv) > 1 else None))
