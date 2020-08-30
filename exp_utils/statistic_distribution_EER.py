from scipy.stats import ttest_rel
import json
import pandas as pd
import random
import argparse
import numpy as np
from sklearn.metrics import roc_curve

DEST_STATISTIC_RES = '/home/meddameloni/dl-fair-voice/exp/statistics/statistic_distribution_EER.csv'
DEST_STATISTIC_FAR = '/home/meddameloni/dl-fair-voice/exp/statistics/statistic_distribution_FAR.csv'
DEST_STATISTIC_FRR = '/home/meddameloni/dl-fair-voice/exp/statistics/statistic_distribution_FRR.csv'

def calculate_threshold(y, y_score):
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
    threshold = thresholds[min_index]
    return threshold


def drop_indices(idx_list, data):
    for idx in idx_list:
        data = data.drop(index=idx, axis=0)
    return data

def countFRR(gbu_data, thr):
    gbu_res = {}
    gbu_size = gbu_data.size()
    frr_res_users = []
    for name, group in gbu_data:
        for index, row in group.iterrows():
            if (row['simlarity'] < thr):
                if (name in gbu_res.keys()):
                    gbu_res[name] += 1
                else:
                    gbu_res[name] = 1
            else:
                if (name in gbu_res):
                    continue
                else:
                    gbu_res[name] = 0
    for user in gbu_res.keys():
        frr_res_users.append(gbu_res[user]/gbu_size[user])

    return sum(frr_res_users)/len(frr_res_users), frr_res_users


def countFAR(gbu_data, thr):
    gbu_res = {}
    gbu_size = gbu_data.size()
    frr_res_users = []
    for name, group in gbu_data:
        for index, row in group.iterrows():
            if (row['simlarity'] > thr):
                if (name in gbu_res.keys()):
                    gbu_res[name] += 1
                else:
                    gbu_res[name] = 1
            else:
                if (name in gbu_res):
                    continue
                else:
                    gbu_res[name] = 0
    for user in gbu_res.keys():
        frr_res_users.append(gbu_res[user] / gbu_size[user])

    return sum(frr_res_users) / len(frr_res_users), frr_res_users


def computeResultsFAR(path):
    data = pd.read_csv(path)

    # split column 'audio_1' in two, in order to group by user
    data[['user_1', 'audio_1']] = data['audio_1'].str.rsplit('/', 1, expand=True)

    label = data.loc[:, 'label']
    similarity = data.loc[:, 'simlarity']

    thr = calculate_threshold(label, similarity)

    false_labels = data[label == 0].reset_index()

    false_labels_m = false_labels[false_labels.gender_1 == 'male']
    false_labels_f = false_labels[false_labels.gender_1 == 'female']
    false_labels_y = false_labels[false_labels.age_1 == 'young']
    false_labels_o = false_labels[false_labels.age_1 == 'old']

    tl_gbu_m = false_labels_m.groupby('user_1', axis=0)
    tl_gbu_f = false_labels_f.groupby('user_1', axis=0)
    tl_gbu_y = false_labels_y.groupby('user_1', axis=0)
    tl_gbu_o = false_labels_o.groupby('user_1', axis=0)

    far_avg_m, far_avg_m_users = countFAR(tl_gbu_m, thr)
    far_avg_f, far_avg_f_users = countFAR(tl_gbu_f, thr)
    far_avg_y, far_avg_y_users = countFAR(tl_gbu_y, thr)
    far_avg_o, far_avg_o_users = countFAR(tl_gbu_o, thr)

    print('Male Users: {}\tFemale Users: {}'.format(len(far_avg_m_users), len(far_avg_f_users)))
    print('Young Users: {}\tOld Users: {}'.format(len(far_avg_y_users), len(far_avg_o_users)))

    res_mf = t_test(far_avg_m_users, far_avg_f_users)
    res_yo = t_test(far_avg_y_users, far_avg_o_users)

    return far_avg_m, far_avg_f, far_avg_y, far_avg_o, res_mf, res_yo


def computeResultsFRR(path):
    data = pd.read_csv(path)

    # split column 'audio_1' in two, in order to group by user
    data[['user_1', 'audio_1']] = data['audio_1'].str.rsplit('/', 1, expand=True)

    label = data.loc[:, 'label']
    similarity = data.loc[:, 'simlarity']

    thr = calculate_threshold(label, similarity)

    true_labels = data[label == 1].reset_index()

    true_labels_m = true_labels[true_labels.gender_1 == 'male']
    true_labels_f = true_labels[true_labels.gender_1 == 'female']
    true_labels_y = true_labels[true_labels.age_1 == 'young']
    true_labels_o = true_labels[true_labels.age_1 == 'old']

    tl_gbu_m = true_labels_m.groupby('user_1', axis=0)
    tl_gbu_f = true_labels_f.groupby('user_1', axis=0)
    tl_gbu_y = true_labels_y.groupby('user_1', axis=0)
    tl_gbu_o = true_labels_o.groupby('user_1', axis=0)

    far_avg_m, far_avg_m_users = countFRR(tl_gbu_m, thr)
    far_avg_f, far_avg_f_users = countFRR(tl_gbu_f, thr)
    far_avg_y, far_avg_y_users = countFRR(tl_gbu_y, thr)
    far_avg_o, far_avg_o_users = countFRR(tl_gbu_o, thr)

    print('Male Users: {}\tFemale Users: {}'.format(len(far_avg_m_users), len(far_avg_f_users)))
    print('Young Users: {}\tOld Users: {}'.format(len(far_avg_y_users), len(far_avg_o_users)))

    res_mf = t_test(far_avg_m_users, far_avg_f_users)
    res_yo = t_test(far_avg_y_users, far_avg_o_users)

    return far_avg_m, far_avg_f, far_avg_y, far_avg_o, res_mf, res_yo


def computeResultsSimilarity(path):
    data = pd.read_csv(path)

    #split column 'audio_1' in two, in order to group by user
    data[['user_1', 'audio_1']] = data['audio_1'].str.rsplit('/', 1, expand=True)

    #creating views for taking into account similarity score divided by categories
    data_m = data[data.gender_1 == 'male'].reset_index()
    data_f = data[data.gender_1 == 'female'].reset_index()
    data_y = data[data.age_1 == 'young'].reset_index()
    data_o = data[data.age_1 == 'old'].reset_index()

    #calculate the distance between rows in related groups
    dist_mf = len(data_m) - len(data_f)
    dist_yo = len(data_y) - len(data_o)

    #if the length of the related groups are not equals
    if (dist_mf !=0 or dist_yo != 0):

        print('difference M/F: {}\tdifference Y/O: {}'.format(dist_mf, dist_yo))
        #PROCESS FOR REDUCE INSTANCES FOR MALE AND FEMALE GROUPS
        #make a view for the related categories group by user ID
        gbu_m = data_m.groupby('user_1', axis=0)
        gbu_f = data_f.groupby('user_1', axis=0)

        # if the difference is greater than 0 we're going to drop rows from male set
        # otherwise we're going to drop rows from female set
        choice_group_to_reduce = ''
        list_idx_to_drop = []
        if dist_mf > 0:
            choice_group_to_reduce = 'm'
        elif dist_mf < 0:
            choice_group_to_reduce = 'f'

        #since the distance is different from 0
        while (dist_mf != 0):
            if (dist_mf > 0):
                for gr in gbu_m.indices.keys():
                    if (dist_mf == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_m.indices[gr])-1)
                        if (gbu_m.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_m.indices[gr][idx_to_drop])
                            check = False
                    dist_mf -= 1
            elif (dist_mf < 0):
                for gr in gbu_f.indices.keys():
                    if (dist_mf == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_f.indices[gr]) - 1)
                        if (gbu_f.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_f.indices[gr][idx_to_drop])
                            check = False
                    dist_mf += 1
            else:
                break

        if choice_group_to_reduce == 'm':
            data_m = drop_indices(list_idx_to_drop, data_m)
            print('> Male set reduced at {} rows'.format(len(data_m)))
        elif choice_group_to_reduce == 'f':
            data_f = drop_indices(list_idx_to_drop, data_f)
            print('> Female set reduced at {} rows'.format(len(data_f)))

        gbu_y = data_y.groupby('user_1', axis=0)
        gbu_o = data_o.groupby('user_1', axis=0)

        choice_group_to_reduce = ''
        list_idx_to_drop = []
        if dist_yo > 0:
            choice_group_to_reduce = 'y'
        elif dist_yo < 0:
            choice_group_to_reduce = 'o'

        while (dist_yo != 0):
            if (dist_yo > 0):
                for gr in gbu_y.indices.keys():
                    if (dist_yo == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_y.indices[gr]) - 1)
                        if (gbu_y.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_y.indices[gr][idx_to_drop])
                            check = False
                    dist_yo -= 1
            elif (dist_yo < 0):
                for gr in gbu_o.indices.keys():
                    if (dist_yo == 0):
                        break
                    check = True
                    while (check):
                        idx_to_drop = random.randint(0, len(gbu_o.indices[gr]) - 1)
                        if (gbu_o.indices[gr][idx_to_drop] not in list_idx_to_drop):
                            list_idx_to_drop.append(gbu_o.indices[gr][idx_to_drop])
                            check = False
                    dist_yo += 1
            else:
                break

        if choice_group_to_reduce == 'y':
            data_y = drop_indices(list_idx_to_drop, data_y)
            print('> Young set reduced at {} rows'.format(len(data_y)))
        elif choice_group_to_reduce == 'o':
            data_o = drop_indices(list_idx_to_drop, data_o)
            print('> Old set reduced at {} rows'.format(len(data_o)))

    similarity_m = data_m.loc[:, 'simlarity']
    similarity_f = data_f.loc[:, 'simlarity']
    similarity_y = data_y.loc[:, 'simlarity']
    similarity_o = data_o.loc[:, 'simlarity']

    print('MALE: {} | FEMALE: {}\tYOUNG: {} | OLD: {}'.format(len(similarity_m), len(similarity_f), len(similarity_y), len(similarity_o)))

    return t_test(similarity_m, similarity_f), t_test(similarity_y, similarity_o)


def t_test(data1, data2, return_verbose=True):
    stat, p = ttest_rel(data1, data2)
    stats_str = 'stat={0:.3f}, p={1:.3f}'.format(stat, p)
    print(stats_str)
    if p > 0.05:
        return 'Y'
    else:
        return 'N'

    # if return_verbose:
    #     return stats_str, p_str


def main():
    parser = argparse.ArgumentParser(description='Operations utils for test results elaboration')

    parser.add_argument('--eer', dest='eer_mode', default=True, type=bool,
                        action='store', help='Base path for results')
    parser.add_argument('--far', dest='far_mode', default=False, type=bool,
                        action='store', help='Base path for destination folder results')
    parser.add_argument('--frr', dest='frr_mode', default=False, type=bool,
                        action='store', help='Base path for destination folder results')

    args = parser.parse_args()

    if args.far_mode == True or args.frr_mode == True:
        args.eer_mode = False

    if (args.eer_mode):
        columns = ['test file', 'network', 'train file', 'accuracy', 'distribution M/F', 'distribution Y/O']
        recordsToLoad = []
        with open('path_best_results.json') as json_file:
            br = json.load(json_file)

            for tests in br.keys():
                print('>  Elaborating results for {}'.format(tests))
                for test_path in br[tests]:
                    res_test_namefile = test_path.split('/')[-1]
                    print('>  Compute distributions for {}'.format(res_test_namefile.split('_')[1]))

                    res_mf, res_yo = computeResultsSimilarity(test_path)
                    recordToAdd = [
                        tests,
                        res_test_namefile.split('_')[0],
                        res_test_namefile.split('_')[1],
                        res_test_namefile.split('_')[2],
                        res_mf,
                        res_yo
                    ]
                    recordsToLoad.append(recordToAdd)

        df = pd.DataFrame(recordsToLoad)
        df.columns = columns
        df.to_csv(DEST_STATISTIC_RES, index=False)
    elif (args.far_mode):
        print('FAR MODE ON')
        columns = ['test file',
                   'network',
                   'train file',
                   'accuracy',
                   'count M FAR',
                   'count F FAR',
                   'count Y FAR',
                   'count O FAR',
                   'Stats correl M/F',
                   'Stats correl Y/O']
        recordsToLoad = []
        with open('path_best_results.json') as json_file:
            br = json.load(json_file)

            for tests in br.keys():
                print('>  Elaborating results for {}'.format(tests))
                for test_path in br[tests]:
                    res_test_namefile = test_path.split('/')[-1]
                    print('>  Compute distributions for {}'.format(res_test_namefile.split('_')[1]))
                    count_m, count_f, count_y, count_o, res_stat_mf, res_stat_yo = computeResultsFAR(test_path)
                    recordToAdd = [
                        tests,
                        res_test_namefile.split('_')[0],
                        res_test_namefile.split('_')[1],
                        res_test_namefile.split('_')[2],
                        count_m,
                        count_f,
                        count_y,
                        count_o,
                        res_stat_mf,
                        res_stat_yo
                    ]
                    recordsToLoad.append(recordToAdd)
        df = pd.DataFrame(recordsToLoad)
        df.columns = columns
        df.to_csv(DEST_STATISTIC_FAR, index=False)

    elif args.frr_mode:
        print('FRR MODE ON')
        columns = ['test file',
                   'network',
                   'train file',
                   'accuracy',
                   'count M FRR',
                   'count F FRR',
                   'count Y FRR',
                   'count O FRR',
                   'Stats correl M/F FRR',
                   'Stats correl Y/O FRR']
        recordsToLoad = []
        with open('path_best_results.json') as json_file:
            br = json.load(json_file)

            for tests in br.keys():
                print('>  Elaborating results for {}'.format(tests))
                for test_path in br[tests]:
                    res_test_namefile = test_path.split('/')[-1]
                    print('>  Compute distributions for {}'.format(res_test_namefile.split('_')[1]))
                    count_m, count_f, count_y, count_o, res_stat_mf, res_stat_yo = computeResultsFRR(test_path)
                    recordToAdd = [
                        tests,
                        res_test_namefile.split('_')[0],
                        res_test_namefile.split('_')[1],
                        res_test_namefile.split('_')[2],
                        count_m,
                        count_f,
                        count_y,
                        count_o,
                        res_stat_mf,
                        res_stat_yo
                    ]
                    recordsToLoad.append(recordToAdd)
        df = pd.DataFrame(recordsToLoad)
        df.columns = columns
        df.to_csv(DEST_STATISTIC_FRR, index=False)

if __name__ == "__main__":
   main()
