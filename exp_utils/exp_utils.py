import os
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import argparse
import shutil
from collections import Counter


RESULT_PATH = '/home/meddameloni/dl-fair-voice/exp/results/'
DEST_PATH = '/home/meddameloni/dl-fair-voice/exp/metrics'


def calculate_parameters(y, y_score):
    """
    Function to calcolate EER, FAR, FRR
    :param y:               real result
    :param y_score:         predicted result
    :return:                min_index
                            FAR
                            FRR
                            FAR (min index)
                            FRR (min index)
                            EER
                            Threshold
                            Thresholds
    """

    far, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    frr = 1 - tpr
    abs_diffs = np.abs(far - frr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((far[min_index], frr[min_index]))
    thresh = thresholds[min_index]
    return min_index, far, frr, far[min_index], frr[min_index], eer, thresh, thresholds


def load_results(data, res_filename):

    # All
    label = data.loc[:, 'label']
    similarity = data.loc[:, 'simlarity']
    m, far_t, frr_t, far, frr, eer, thr, thresholds = calculate_parameters(label, similarity)

    # Old
    # data_o = data[data.age_1 == 'old']
    # similarity_o = data_o.loc[:, 'simlarity']
    # label_o = data_o.loc[:, 'label']
    # m_o, far_ot, frr_ot, far_o, frr_o, eer_o, thr_o, t_o = calculate_parameters(label_o, similarity_o)

    # Young
    # data_y = data[data.age_1 == 'young']
    # similarity_y = data_y.loc[:, 'simlarity']
    # label_y = data_y.loc[:, 'label']
    # m_y, far_yt, frr_yt, far_y, frr_y, eer_y, thr_y, t_y = calculate_parameters(label_y, similarity_y)

    # Female
    # data_fm = data[data.gender_1 == 'female']
    # similarity_fm = data_fm.loc[:, 'simlarity']
    # label_fm = data_fm.loc[:, 'label']
    # m_fm, far_fmt, frr_fmt, far_fm, frr_fm, eer_fm, thr_fm, t_fm = calculate_parameters(label_fm, similarity_fm)

    # Male
    # data_ml = data[data.gender_1 == 'male']
    # similarity_ml = data_ml.loc[:, 'simlarity']
    # label_ml = data_ml.loc[:, 'label']
    # m_ml, far_mlt, frr_mlt, far_ml, frr_ml, eer_ml, thr_ml, t_ml = calculate_parameters(label_ml, similarity_ml)

    err = round(eer * 100, 2)
    # err_o = round(eer_o * 100, 2)
    # err_y = round(eer_y * 100, 2)
    # err_fm = round(eer_fm * 100, 2)
    # err_ml = round(eer_ml * 100, 2)
    #
    # far_o = round(far_o * 100, 2)
    # far_y = round(far_y * 100, 2)
    # far_fm = round(far_fm * 100, 2)
    # far_ml = round(far_ml * 100, 2)
    far = round(far * 100, 2)

    # frr_o = round(frr_o * 100, 2)
    # frr_y = round(frr_y * 100, 2)
    # frr_fm = round(frr_fm * 100, 2)
    # frr_ml = round(frr_ml * 100, 2)
    frr = round(frr * 100, 2)

    # err_record = [res_filename.split('_')[0], res_filename.split('_')[1], res_filename.split('_')[4],  res_filename.split('_')[2][:2] + '.' + res_filename.split('_')[2][-1], err, err_o, err_y, err_fm, err_ml]
    # far_record = [res_filename.split('_')[0], res_filename.split('_')[1], res_filename.split('_')[4],  res_filename.split('_')[2][:2] + '.' + res_filename.split('_')[2][-1], far, far_o, far_y, far_fm, far_ml]
    # frr_record = [res_filename.split('_')[0], res_filename.split('_')[1], res_filename.split('_')[4],  res_filename.split('_')[2][:2] + '.' + res_filename.split('_')[2][-1], frr, frr_o, frr_y, frr_fm, frr_ml]

    # Unlock if interested only in total amount parameters
    err_record = [res_filename.split('_')[0], res_filename.split('_')[1], res_filename.split('_')[4],
                  res_filename.split('_')[2][:2] + '.' + res_filename.split('_')[2][-1], err]
    far_record = [res_filename.split('_')[0], res_filename.split('_')[1], res_filename.split('_')[4],
                  res_filename.split('_')[2][:2] + '.' + res_filename.split('_')[2][-1], far]
    frr_record = [res_filename.split('_')[0], res_filename.split('_')[1], res_filename.split('_')[4],
                  res_filename.split('_')[2][:2] + '.' + res_filename.split('_')[2][-1], frr]

    return err_record, far_record, frr_record


def create_Experiment_CSV_details(eer, far, frr, dest_path):
    if (not os.path.exists(dest_path)):
        os.mkdir(dest_path)
    else:
        shutil.rmtree(dest_path)
        os.mkdir(dest_path)

    df_eer = pd.DataFrame(eer)
    df_eer.columns = ['Architecture', 'Train File', 'Test File', 'Accuracy', 'EER', 'EER Old', 'EER Young', 'EER Female', 'EER Male']
    eer_csv_name = 'EER' + '.csv'
    eer_path = os.path.join(dest_path, eer_csv_name)
    df_eer.to_csv(eer_path, index=False)
    print('> EER CSV GENERATED in \t' + dest_path)

    df_far = pd.DataFrame(far)
    df_far.columns = ['Architecture', 'Train File', 'Test File', 'Accuracy', 'FAR', 'FAR Old', 'FAR Young', 'FAR Female', 'FAR Male']
    far_csv_name = 'FAR' + '.csv'
    df_far.to_csv(os.path.join(dest_path, far_csv_name), index=False)
    print('> FAR CSV GENERATED in \t' + dest_path)

    df_frr = pd.DataFrame(frr)
    df_frr.columns = ['Architecture', 'Train File', 'Test File', 'Accuracy', 'FRR', 'FRR Old', 'FRR Young', 'FRR Female', 'FRR Male']
    frr_csv_name = 'FRR' + '.csv'
    df_frr.to_csv(os.path.join(dest_path, frr_csv_name), index=False)
    print('> FRR CSV GENERATED in \t' + dest_path)


def create_Experiment_CSV_details_totEERonly(eer, far, frr, dest_path):

    if (not os.path.exists(dest_path)):
        os.mkdir(dest_path)
    else:
        shutil.rmtree(dest_path)
        os.mkdir(dest_path)

    df_eer = pd.DataFrame(eer)
    df_eer.columns = ['Architecture', 'Train File', 'Test File', 'Accuracy', 'EER']
    eer_csv_name = 'EER' + '.csv'
    eer_path = os.path.join(dest_path, eer_csv_name)
    df_eer.to_csv(eer_path, index=False)
    print('> EER CSV GENERATED in \t' + dest_path)

    df_far = pd.DataFrame(far)
    df_far.columns = ['Architecture', 'Train File', 'Test File', 'Accuracy', 'FAR']
    far_csv_name = 'FAR' + '.csv'
    df_far.to_csv(os.path.join(dest_path, far_csv_name), index=False)
    print('> FAR CSV GENERATED in \t' + dest_path)

    df_frr = pd.DataFrame(frr)
    df_frr.columns = ['Architecture', 'Train File', 'Test File', 'Accuracy', 'FRR']
    frr_csv_name = 'FRR' + '.csv'
    df_frr.to_csv(os.path.join(dest_path, frr_csv_name), index=False)
    print('> FRR CSV GENERATED in \t' + dest_path)


def main():
    parser = argparse.ArgumentParser(description='Operations utils for test results elaboration')

    parser.add_argument('--result_path', dest='result_path', default=RESULT_PATH, type=str,
                        action='store', help='Base path for results')
    parser.add_argument('--dest_folder', dest='dest_folder', default=DEST_PATH, type=str,
                        action='store', help='Base path for destination folder results')

    args = parser.parse_args()

    eer_to_load, far_to_load, frr_to_load = [], [], []
    count_res = 0
    print('>Start Scanning Results folder')
    for res in os.listdir(args.result_path):
        if not os.path.isdir((os.path.join(args.result_path, res))):
            count_res += 1
            print('>Elaborating ---> \t' + res.replace('_meta_metadata_test_', ''))
            res_csv = pd.read_csv(os.path.join(args.result_path, res))
            eer, far, frr = load_results(res_csv, res)

            eer_to_load.append(eer)
            far_to_load.append(far)
            frr_to_load.append(frr)

    # create_Experiment_CSV_details(eer_to_load, far_to_load, frr_to_load, args.dest_folder)
    create_Experiment_CSV_details_totEERonly(eer_to_load, far_to_load, frr_to_load, args.dest_folder)
    print('\n\n> {} RESULTS ELABORATED!'.format(count_res))

if __name__ == '__main__':
    main()