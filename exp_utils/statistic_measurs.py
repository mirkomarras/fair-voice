import scipy.stats
import itertools
import os
import pandas as pd
import sys

out_path = 'fair-voice/exp/statistics'


def sample_info_statistics(path):
    """
    It evaluates the statistical relation between the audio files of each group where each group can be one of
    female_old, female_young, male_old, male_young with "paired T-test". It creates a txt file specifying if
    the audio files of the two groups have the same distribution or not.

    :param path: path of the "sample_info" file, e.g. "sample_info_English-test1.csv" file is an accepted file
    :return:
    """
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    df_group = df.groupby('group')
    gr_comb = itertools.combinations(df_group.groups, 2)
    gr_comb = [comb for comb in gr_comb]
    df_stats_cols = list(df.loc[:, "n_sample":"noise_avg"].columns)

    out_cols = ["stats_value", "p_result"]
    multi_index_len = len(df_stats_cols)
    # Creation of a multiindex in the format (1# Level: [(female_old, female_young)],
    #                                         2# Level: [n_sample, duration_avg, min_duration, .....]
    # To create this type of index pandas accepts two arrays, the first array must contain the same value as many as
    # the length of the second array.
    # The first array is created generating an array of length "multi_index_len" for each value of "comb" with "map",
    # then a "chain" is created with itertools and finally the elements of the chain are regrouped in an array
    out_multi_index = pd.MultiIndex.from_arrays([
        list(itertools.chain.from_iterable(map(lambda comb: [comb]*multi_index_len,
                                               gr_comb)
                                           )),
        df_stats_cols * len(gr_comb)
    ])

    data = []
    for grs in gr_comb:
        gr1 = df_group.get_group(grs[0])
        gr2 = df_group.get_group(grs[1])

        for col in df_stats_cols:
            stats_str, p_str = t_test(gr1[col].to_numpy(), gr2[col].to_numpy(), return_verbose=True)
            data.append([stats_str, p_str])

    out_df = pd.DataFrame(data, columns=out_cols, index=out_multi_index)

    with open(os.path.join(out_path, os.path.splitext(os.path.basename(path))[0] + '_statistics.txt'), 'w') as out_file:
        out_file.write(out_df.to_string())


# Paired Student’s t-test code copied from
# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
def t_test(data1, data2, return_verbose=True):
    stat, p = scipy.stats.ttest_rel(data1, data2)
    stats_str = 'stat={0:.3f}, p={1:.3f}'.format(stat, p)
    print(stats_str)
    if p > 0.05:
        p_str = 'Probably the same distribution'
        print(p_str)
    else:
        p_str = 'Probably different distributions'
        print(p_str)

    if return_verbose:
        return stats_str, p_str


if __name__ == "__main__":
    sample_info_statistics(sys.argv[1])
