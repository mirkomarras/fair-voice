import pandas as pd
import os
import shutil
import math
import re
import warnings
import ast
import itertools
import random
from random import shuffle
from collections import defaultdict
from enum import Enum, auto

ROUND_CORRECTOR = 1e-9


class IdCounter:
    def __init__(self, id_start=0):
        self.id = id_start

    @staticmethod
    def get_id_counter_metadata(metadata, folder):
        metas = [os.path.splitext(x)[0] for x in os.listdir(folder) if x.startswith(os.path.splitext(metadata)[0])]
        metas = map(int, filter(None, map(lambda x: x.replace(os.path.splitext(metadata)[0], ''), metas)))
        return IdCounter(max(metas, default=0))

    def up(self):
        self.id += 1
        return self.id


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


# distinct: create a csv for each of the value in group_by_lang, female_male_ratio, (young_old_ratio and young_cap)
#       metadata_1.csv = female_male_ratio[0]
#       ...
#       metadata_n.csv = group_by_lang[0]
#       metadata_n+1.csv = young_old_ratio[0] for each cap in young_cap
#         example with young_old_ratio = [(0.3, 0.7)] and young_cap = ["fourties", "fifties"]
#           metadata_1.csv = labels with (0.3, 0.7) young old ratio where "young" is each row with age < "fourties"
#           metadata_2.csv = labels with (0.3, 0.7) young old ratio where "young" is each row with age < "fifties"
# merge: create a csv for each of the value of the group (female_male_ratio, (young_old_ratio and young_cap)) with
#        the entries of all languages in group_by_lang
#       metadata_1.csv = all langs in group_by_lang, (female_male_ratio[0], (young_old_ratio[0] and young_cap[0]))
#       ...
#       metadata_n.csv = all langs in group_by_lang, (female_male_ratio[0], (young_old_ratio[0] and young_cap[0]))
#       metadata_n+1.csv = all langs in group_by_lang, (female_male_ratio[0], (young_old_ratio[0] and young_cap[1]))
#       metadata_n+m.csv = all langs in group_by_lang, (female_male_ratio[3], (young_old_ratio[1] and young_cap[1]))
# merge_foreach_lang: same behaviour of merge, but taking into account one language at a time.
#                      So if "merge" creates N csvs, "merge_foreach_lang" creates N*len(group_by_lang) csvs
# distinct_merge: shortcut to do both merge and distinct at once
#
# distinct_merge_foreach_lang: shortcut to do both merge_foreach_lang and distinct at once
#
# -#-# for different behaviours you can set to None some values.
#       Example: if you want to get the labels of all the languages with a young_old_ratio of (0.7, 0.3) and
#                young_cap = "sixties" you can call the method in this way:
#                def prepare_common_voice(....other_params....,
#                                         young_cap=["sixties"],
#                                         female_male_ratio=None,
#                                         young_old_ratio=[(0.7, 0.3)])
#
class Modes(AutoName):
    DISTINCT = auto()
    MERGE = auto()
    MERGE_FOREACH_LANG = auto()
    DISTINCT_MERGE = auto()
    DISTINCT_MERGE_FOREACH_LANG = auto()

    @staticmethod
    def check_mode(mode):
        if isinstance(mode, str):
            if mode not in [_mode.value for _mode in Modes.__members__.values()]:
                raise ValueError(
                    '"mode" can only have the values {}'.format([x.value for x in Modes.__members__.values()]))
        elif not isinstance(mode, Modes):
            raise ValueError(
                '"mode" can only have the values {}'.format([x.value for x in Modes.__members__.values()]))

    @staticmethod
    def is_mode_instance(o, *args):
        is_mode = False
        for mode in args:
            is_mode |= o == mode.value or o == mode

        return is_mode


class TestUsersLoader:
    def __init__(self, path):
        self.path = path
        if os.path.exists(self.path):
            with open(self.path, 'r') as test_users_file:
                self.test_users = ast.literal_eval(test_users_file.read())
        else:
            self.test_users = []

    def save(self):
        with open(self.path, 'w') as test_users_file:
            test_users_file.write(str(list(self.test_users)))


ages_vals = [
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
    "nineties"
]

default = object()


def prepare_common_voice(metadata="metadata.csv",
                         encoding="utf-8",
                         dir_metadata="/home/meddameloni/FairVoice/metadata",
                         young_cap=default,
                         min_sample=5,
                         group_by_lang=default,
                         lang_equality=False,
                         female_male_ratio=default,
                         young_old_ratio=default,
                         mode=Modes.MERGE,
                         tot_test_users=100):
    metadata_id = IdCounter.get_id_counter_metadata(metadata, dir_metadata)
    start_id = metadata_id.id

    csvs_info = {
        "name": [],
        "group by language": [],
        "female male ratio": [],
        "young old ratio": [],
        "young cap": [],
        "n_sample": [],
        "mode": []
    }

    if young_cap is default or not young_cap:
        young_cap = ["fourties"]

    if female_male_ratio is default:
        female_male_ratio = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4)]

    if young_old_ratio is default:
        young_old_ratio = [(0.4, 0.6), (0.5, 0.5), (0.6, 0.4)]

    Modes.check_mode(mode)

    labels = pd.read_csv(os.path.join(dir_metadata, metadata), sep=',', encoding=encoding)
    labels = labels.loc[labels["n_sample"] >= min_sample]

    languages = labels.groupby("language_l1")
    if group_by_lang is default:
        groups = [lang for lang in languages.groups]
    elif group_by_lang:
        groups = group_by_lang
        group_errors = [lang for lang in groups if lang not in languages.groups]
        if group_errors:
            raise ValueError('{} are not languages of the dataset', group_errors)
    else:
        groups = None

    if lang_equality and group_by_lang:
        min_lang_users = min([len(languages.get_group(lang)) for lang in groups])

    if Modes.is_mode_instance(mode, Modes.DISTINCT):
        # if lang_equality:
        #     warnings.warn("lang_equality is pointless in DISTINCT mode", RuntimeWarning)

        prepare_distinct(metadata=metadata,
                         dir_metadata=dir_metadata,
                         labels=labels,
                         csvs_info=csvs_info,
                         metadata_id=metadata_id,
                         groups=groups,
                         female_male_ratio=female_male_ratio,
                         young_old_ratio=young_old_ratio,
                         young_cap=young_cap,
                         mode=mode,
                         min_lang_users_test=(min_lang_users, tot_test_users) if "min_lang_users" in locals() else None)

    else:
        # at least one among "female_male_ratio", "young_old_ratio" needs to be an array of values (Nothing to merge)
        if not (female_male_ratio or young_old_ratio):
            raise ValueError('female_male_ratio or young_old_ratio must not be empty. Nothing to merge')

        if not groups and not (female_male_ratio and young_old_ratio):
            raise ValueError('groups = None is possible if both female_male_ratio and young_old_ratio are valid, '
                             'use DISTINCT mode instead')

        prepare_merge(metadata=metadata,
                      dir_metadata=dir_metadata,
                      labels=labels,
                      csvs_info=csvs_info,
                      metadata_id=metadata_id,
                      groups=groups,
                      female_male_ratio=female_male_ratio,
                      young_old_ratio=young_old_ratio,
                      young_cap=young_cap,
                      mode=mode,
                      min_lang_users=min_lang_users if "min_lang_users" in locals() else None)

        if Modes.is_mode_instance(mode, Modes.DISTINCT_MERGE, Modes.DISTINCT_MERGE_FOREACH_LANG):
            prepare_distinct(metadata=metadata,
                             dir_metadata=dir_metadata,
                             labels=labels,
                             csvs_info=csvs_info,
                             metadata_id=metadata_id,
                             groups=groups,
                             female_male_ratio=female_male_ratio,
                             young_old_ratio=young_old_ratio,
                             young_cap=young_cap,
                             mode=mode)

    save_info(csvs_info, dir_metadata, start_id, metadata_id.id)

    return start_id + 1


def prepare_distinct(metadata=None,
                     dir_metadata=None,
                     labels=None,
                     csvs_info=None,
                     metadata_id=None,
                     groups=None,
                     female_male_ratio=None,
                     young_old_ratio=None,
                     young_cap=None,
                     mode=None,
                     min_lang_users_test=None):
    local_mode = "{} ({})".format(mode.value, "distinct")

    if groups:
        for lang in groups:
            languages = labels.groupby("language_l1")
            lang_df = languages.get_group(lang)

            if min_lang_users_test:
                loaded_test_users = TestUsersLoader(
                    os.path.join(dir_metadata, 'test_users_{}_{}'.format(lang, min_lang_users_test[1]))
                )
                _, id_users = zip(*loaded_test_users.test_users)
                lang_df_1 = lang_df[lang_df["id_user"].isin(id_users)]
                lang_df_2 = lang_df[~lang_df["id_user"].isin(id_users)]
                lang_df_2 = lang_df_2.sort_values("n_sample", ascending=False).\
                    head(min_lang_users_test[0] - min_lang_users_test[1])
                lang_df = pd.concat([lang_df_1, lang_df_2])

            out_name = save_metadata(metadata, lang_df, dir_metadata, metadata_id)

            update_info(csvs_info,
                        out_name,
                        '{}->{}'.format(groups, lang),
                        None,
                        None,
                        None,
                        len(lang_df),
                        local_mode)

    if female_male_ratio:
        for fm_ratio in female_male_ratio:
            f_ratio = fm_ratio[0]
            m_ratio = fm_ratio[1]

            assert math.isclose(f_ratio + m_ratio, 1.0), "The sum of each tuple in female_male_ratio must be 1.0"

            genders = labels.groupby("gender")

            females = genders.get_group("female")
            males = genders.get_group("male")

            # tuple that associate a group and its new length
            excess = get_excess("male", "female", males, females, m_ratio, f_ratio, len(labels))

            if excess[0] == "male":
                males = males.sort_values("n_sample", ascending=False).head(excess[1])
            else:
                females = females.sort_values("n_sample", ascending=False).head(excess[1])

            fm_df = pd.concat([females, males])

            out_name = save_metadata(metadata, fm_df, dir_metadata, metadata_id)

            update_info(csvs_info,
                        out_name,
                        None,
                        fm_ratio,
                        None,
                        None,
                        len(fm_df),
                        local_mode)

    if young_old_ratio:
        for y_cap in young_cap:
            ages = {k: "young" if ages_vals.index(y_cap) > i else "old" for i, k in enumerate(ages_vals)}
            for yo_ratio in young_old_ratio:
                y_ratio = yo_ratio[0]
                o_ratio = yo_ratio[1]

                assert math.isclose(y_ratio + o_ratio, 1.0), "The sum of each tuple in young_old_ratio must be 1.0"

                group_ages = labels.set_index("age").groupby(ages)

                youngs = group_ages.get_group("young").reset_index()
                olds = group_ages.get_group("old").reset_index()

                excess = get_excess("old", "young", olds, youngs, o_ratio, y_ratio, len(labels))

                if excess[0] == "old":
                    olds = olds.sort_values("n_sample", ascending=False).head(excess[1])
                else:
                    youngs = youngs.sort_values("n_sample", ascending=False).head(excess[1])

                yo_df = pd.concat([youngs, olds])
                yo_df.insert(5, "age", yo_df.pop("age"))

                out_name = save_metadata(metadata, yo_df, dir_metadata, metadata_id)

                update_info(csvs_info,
                            out_name,
                            None,
                            None,
                            yo_ratio,
                            y_cap,
                            len(yo_df),
                            local_mode)


def prepare_merge(metadata=None,
                  dir_metadata=None,
                  labels=None,
                  csvs_info=None,
                  metadata_id=None,
                  groups=None,
                  female_male_ratio=None,
                  young_old_ratio=None,
                  young_cap=None,
                  mode=None,
                  min_lang_users=None):
    local_mode = "{} ({})".format(mode.value, "merge")
    tot_metadata = defaultdict(pd.DataFrame)

    if groups:
        languages = labels.groupby("language_l1")

        # used for language equality
        if min_lang_users is not None:
            out_metadata = []

            # languages = pd.concat([languages.get_group(lang).
            #                       sort_values("n_sample", ascending=False).head(min_lang_users) for lang in groups])
            # languages = languages.groupby("language_l1")

        for lang in groups:
            rows = len(languages.get_group(lang))
            process_fm_yo_ratios(metadata,
                                 dir_metadata,
                                 rows,
                                 csvs_info,
                                 metadata_id,
                                 female_male_ratio,
                                 young_old_ratio,
                                 young_cap,
                                 tot_metadata,
                                 languages,
                                 mode,
                                 local_mode,
                                 min_lang_users_data=out_metadata if min_lang_users else None,
                                 lang=lang)

        if min_lang_users is not None and \
                Modes.is_mode_instance(mode, Modes.MERGE_FOREACH_LANG, Modes.DISTINCT_MERGE_FOREACH_LANG):

            min_length = 1e10
            for _, data_groups in out_metadata:
                data_sum = sum([len(d_group) for d_group in data_groups])
                if data_sum < min_length:
                    min_length = data_sum

            n_data_groups = len(out_metadata[0][1])
            min_length = (min_length // n_data_groups) * n_data_groups  # if odd it becomes even
            min_for_group = min_length // n_data_groups

            for _lang, data_groups in out_metadata:
                data_minimized = list(map(lambda df: df.sort_values("n_sample", ascending=False).head(min_for_group),
                                          data_groups))

                final_metadata = pd.concat(data_minimized)
                final_metadata.insert(5, "age", final_metadata.pop("age"))

                out_name = save_metadata(metadata, final_metadata, dir_metadata, metadata_id)

                update_info(csvs_info,
                            out_name,
                            _lang,
                            (0.5, 0.5),
                            (0.5, 0.5),
                            young_cap[0],
                            len(final_metadata),
                            local_mode)

        if Modes.is_mode_instance(mode, Modes.MERGE, Modes.DISTINCT_MERGE):
            for key in tot_metadata:
                fm_ratio = re.search(r'(?<=fm).+?[)]', key)
                fm_ratio = fm_ratio[0] if fm_ratio else None

                yo_ratio = re.search(r'(?<=yo).+?[)]', key)
                yo_ratio = yo_ratio[0] if yo_ratio else None

                y_cap = re.search(r'(?<=[)] ).*', key)
                y_cap = y_cap[0] if y_cap else None

                meta_to_save = tot_metadata[key]

                if fm_ratio == '(0.5, 0.5)' and yo_ratio == '(0.5, 0.5)' and min_lang_users:
                    min_length = 1e10
                    for _, data_groups in out_metadata:
                        data_sum = sum([len(d_group) for d_group in data_groups])
                        if data_sum < min_length:
                            min_length = data_sum

                    n_data_groups = len(out_metadata[0][1])
                    min_length = (min_length // n_data_groups) * n_data_groups
                    min_for_group = min_length // n_data_groups

                    min_lang_meta = None
                    for _lang, data_groups in out_metadata:
                        data_minimized = list(
                            map(lambda df: df.sort_values("n_sample", ascending=False).head(min_for_group),
                                data_groups))

                        lang_meta = pd.concat(data_minimized)
                        lang_meta.insert(5, "age", lang_meta.pop("age"))
                        min_lang_meta = pd.concat([min_lang_meta, lang_meta])

                    meta_to_save = min_lang_meta
                out_name = save_metadata(metadata, meta_to_save, dir_metadata, metadata_id)

                update_info(csvs_info,
                            out_name,
                            groups,
                            fm_ratio,
                            yo_ratio,
                            y_cap,
                            len(tot_metadata[key]),
                            local_mode)

    elif young_old_ratio and female_male_ratio:
        process_fm_yo_ratios(metadata,
                             dir_metadata,
                             len(labels),
                             csvs_info,
                             metadata_id,
                             female_male_ratio,
                             young_old_ratio,
                             young_cap,
                             tot_metadata,
                             labels,
                             mode,
                             local_mode)

        if Modes.is_mode_instance(mode, Modes.MERGE, Modes.DISTINCT_MERGE):
            for key in tot_metadata:
                out_name = save_metadata(metadata, tot_metadata[key], dir_metadata, metadata_id)

                fm_ratio = re.search(r'(?<=fm).+?[)]', key)
                fm_ratio = fm_ratio[0] if fm_ratio else None

                yo_ratio = re.search(r'(?<=yo).+?[)]', key)
                yo_ratio = yo_ratio[0] if yo_ratio else None

                y_cap = re.search(r'(?<=[)] ).*', key)
                y_cap = y_cap[0] if y_cap else None

                update_info(csvs_info,
                            out_name,
                            None,
                            fm_ratio,
                            yo_ratio,
                            y_cap,
                            len(tot_metadata[key]),
                            local_mode)
    else:
        # this line should never be reachable
        raise ValueError('groups = None is possible if both female_male_ratio and young_old_ratio are valid, '
                         'use DISTINCT mode instead')


def process_fm_yo_ratios(metadata,
                         dir_metadata,
                         rows,
                         csvs_info,
                         metadata_id,
                         female_male_ratio,
                         young_old_ratio,
                         young_cap,
                         tot_metadata,
                         labels,
                         mode,
                         local_mode,
                         min_lang_users_data=None,
                         lang=None):
    if female_male_ratio:
        for fm_ratio in female_male_ratio:
            f_ratio = fm_ratio[0]
            m_ratio = fm_ratio[1]

            assert math.isclose(f_ratio + m_ratio, 1.0), "The sum of each tuple in female_male_ratio must be 1.0"

            if lang:
                languages = labels
                genders = languages.get_group(lang).groupby("gender")
            else:
                genders = labels.groupby("gender")

            females = genders.get_group("female")
            males = genders.get_group("male")

            # tuple that associate a group and its new length
            excess = get_excess("male", "female", males, females, m_ratio, f_ratio, rows)

            if young_old_ratio:
                for y_cap in young_cap:
                    ages = {k: "young" if ages_vals.index(y_cap) > i else "old" for i, k in enumerate(ages_vals)}
                    for yo_ratio in young_old_ratio:
                        y_ratio = yo_ratio[0]
                        o_ratio = yo_ratio[1]

                        assert math.isclose(y_ratio + o_ratio, 1.0), \
                            "The sum of each tuple in young_old_ratio must be 1.0"

                        male_by_age = males.set_index("age").groupby(ages)
                        male_youngs = male_by_age.get_group("young").reset_index()
                        male_olds = male_by_age.get_group("old").reset_index()

                        female_by_age = females.set_index("age").groupby(ages)
                        female_youngs = female_by_age.get_group("young").reset_index()
                        female_olds = female_by_age.get_group("old").reset_index()

                        if excess[0] == "male":
                            tot_f = len(female_youngs) + len(female_olds)
                            if len(female_youngs) < round(tot_f * y_ratio - ROUND_CORRECTOR):
                                new_f_olds = round(len(female_youngs) / y_ratio - ROUND_CORRECTOR) - len(female_youngs)
                                female_olds = female_olds.sort_values("n_sample", ascending=False).head(new_f_olds)
                                tot_f = len(female_youngs) + new_f_olds
                            elif len(female_youngs) > round(tot_f * y_ratio - ROUND_CORRECTOR):
                                new_f_youngs = round(len(female_olds) / o_ratio - ROUND_CORRECTOR) - len(female_olds)
                                female_youngs = female_youngs.sort_values("n_sample", ascending=False).head(new_f_youngs)
                                tot_f = len(female_olds) + new_f_youngs

                            excess = (excess[0], round(tot_f / f_ratio - ROUND_CORRECTOR) - tot_f)

                            sample_size = min(len(male_youngs), round(excess[1] * y_ratio - ROUND_CORRECTOR))
                            male_youngs = male_youngs.sort_values("n_sample", ascending=False).head(sample_size)
                            sample_size = min(len(male_olds), round(excess[1] * o_ratio - ROUND_CORRECTOR))
                            male_olds = male_olds.sort_values("n_sample", ascending=False).head(sample_size)
                        else:
                            tot_m = len(male_youngs) + len(male_olds)
                            if len(male_youngs) < round(tot_m * y_ratio - ROUND_CORRECTOR):
                                new_m_olds = round(len(male_youngs) / y_ratio - ROUND_CORRECTOR) - len(male_youngs)
                                male_olds = male_olds.sort_values("n_sample", ascending=False).head(new_m_olds)
                                tot_m = len(male_youngs) + new_m_olds
                            elif len(male_youngs) > round(tot_m * y_ratio - ROUND_CORRECTOR):
                                new_m_youngs = round(len(male_olds) / o_ratio - ROUND_CORRECTOR) - len(male_olds)
                                male_youngs = male_youngs.sort_values("n_sample", ascending=False).head(new_m_youngs)
                                tot_m = len(male_olds) + new_m_youngs

                            excess = (excess[0], round(tot_m / m_ratio - ROUND_CORRECTOR) - tot_m)

                            sample_size = min(len(female_youngs), round(excess[1] * y_ratio - ROUND_CORRECTOR))
                            female_youngs = female_youngs.sort_values("n_sample", ascending=False).head(sample_size)
                            sample_size = min(len(female_olds), round(excess[1] * o_ratio - ROUND_CORRECTOR))
                            female_olds = female_olds.sort_values("n_sample", ascending=False).head(sample_size)

                        if min_lang_users_data is not None and yo_ratio == (0.5, 0.5) and fm_ratio == (0.5, 0.5):
                            min_lang_users_data.append((lang, [female_olds, female_youngs, male_olds, male_youngs]))

                        out_females = pd.concat([female_olds, female_youngs])
                        out_males = pd.concat([male_olds, male_youngs])

                        out_metadata = pd.concat([out_females, out_males])
                        out_metadata.insert(5, "age", out_metadata.pop("age"))
                        tot_metadata["fm" + str(fm_ratio) + "yo" + str(yo_ratio) + " " + y_cap] = pd.concat(
                            [tot_metadata["fm" + str(fm_ratio) + "yo" + str(yo_ratio) + " " + y_cap],
                             out_metadata]
                        )

                        if lang:
                            if Modes.is_mode_instance(mode, Modes.MERGE_FOREACH_LANG,
                                                      Modes.DISTINCT_MERGE_FOREACH_LANG) and min_lang_users_data is None:
                                out_name = save_metadata(metadata, out_metadata, dir_metadata, metadata_id)

                                update_info(csvs_info,
                                            out_name,
                                            lang,
                                            fm_ratio,
                                            yo_ratio,
                                            y_cap,
                                            len(out_metadata),
                                            local_mode)
                        else:
                            raise ValueError('MERGE_FOREACH_LANG and DISTINCT_MERGE_FOREACH_LANG cannot be used with '
                                             'group_by_lang = None')
            else:
                if excess[0] == "male":
                    males = males.sort_values("n_sample", ascending=False).head(excess[1])
                else:
                    females = females.sort_values("n_sample", ascending=False).head(excess[1])

                out_metadata = pd.concat([females, males])
                tot_metadata["fm" + str(fm_ratio)] = pd.concat([tot_metadata["fm" + str(fm_ratio)], out_metadata])

                if Modes.is_mode_instance(mode, Modes.MERGE_FOREACH_LANG,
                                          Modes.DISTINCT_MERGE_FOREACH_LANG):
                    out_name = save_metadata(metadata, out_metadata, dir_metadata, metadata_id)

                    update_info(csvs_info,
                                out_name,
                                lang,
                                fm_ratio,
                                None,
                                None,
                                len(out_metadata),
                                local_mode)
    elif young_old_ratio:
        for y_cap in young_cap:
            ages = {k: "young" if ages_vals.index(y_cap) > i else "old" for i, k in enumerate(ages_vals)}
            for yo_ratio in young_old_ratio:
                y_ratio = yo_ratio[0]
                o_ratio = yo_ratio[1]

                assert math.isclose(y_ratio + o_ratio, 1.0), \
                    "The sum of each tuple in young_old_ratio must be 1.0"

                # no need to check "if lang" because the else branch is executed only if female_male_ratio is None
                languages = labels
                group_ages = languages.get_group(lang).set_index("age").groupby(ages)

                youngs = group_ages.get_group("young").reset_index()
                olds = group_ages.get_group("old").reset_index()

                excess = get_excess("old", "young", olds, youngs, o_ratio, y_ratio, rows)

                if excess[0] == "old":
                    olds = olds.sort_values("n_sample", ascending=False).head(excess[1])
                else:
                    youngs = youngs.sort_values("n_sample", ascending=False).head(excess[1])

                out_metadata = pd.concat([youngs, olds])
                out_metadata.insert(5, "age", out_metadata.pop("age"))
                tot_metadata["yo" + str(yo_ratio) + " " + y_cap] = pd.concat(
                    [tot_metadata["yo" + str(yo_ratio) + " " + y_cap],
                     out_metadata]
                )

                if Modes.is_mode_instance(mode, Modes.MERGE_FOREACH_LANG,
                                          Modes.DISTINCT_MERGE_FOREACH_LANG):
                    out_name = save_metadata(metadata, out_metadata, dir_metadata, metadata_id)

                    update_info(csvs_info,
                                out_name,
                                lang,
                                None,
                                yo_ratio,
                                y_cap,
                                len(out_metadata),
                                local_mode)

    else:
        # this line should never be reachable
        raise ValueError('female_male_ratio or young_old_ratio must not be empty. Nothing to merge')


def get_excess(key1, key2, data1, data2, ratio1, ratio2, len_all_data):
    """
    :return: a tuple (group, len(new_group)) containing the group with more rows than the ratio and the number of rows
    to match the expected ratio
    """
    excess = (key1, len(data1))
    if len(data2) < round(len_all_data * ratio2 - ROUND_CORRECTOR):
        excess = (key1, round(len(data2) / ratio2 - ROUND_CORRECTOR) - len(data2))
    elif len(data2) > round(len_all_data * ratio2):
        excess = (key2, round(len(data1) / ratio1 - ROUND_CORRECTOR) - len(data1))

    return excess


def update_info(info, name, group_by_lang, female_male_ratio, young_old_ratio, young_cap, n_sample, mode_value):
    info["name"].append(name)
    info["group by language"].append(group_by_lang)
    info["female male ratio"].append(female_male_ratio)
    info["young old ratio"].append(young_old_ratio)
    info["young cap"].append(young_cap)
    info["n_sample"].append(n_sample)
    info["mode"].append(mode_value)


def save_metadata(metadata, out_metadata, dir_metadata, metadata_id):
    out_name = "{}{}.csv".format(os.path.splitext(metadata)[0], str(metadata_id.up()))
    out_metadata.to_csv(os.path.join(dir_metadata, out_name),
                        encoding="utf-8",
                        index=False)

    return out_name


def save_info(csvs_info, dir_metadata, start_id, end_id):
    pd.DataFrame(data=csvs_info).to_csv(
        os.path.join(dir_metadata, "info_metadata_{}_{}.csv".format(start_id + 1, end_id)),
        encoding='utf-8',
        index=False
    )


def split_dataset(metadata="metadata.csv",
                  info_start_id=None,
                  encoding="utf-8",
                  dir_metadata="/home/meddameloni/FairVoice/metadata",
                  dir_dataset="/home/meddameloni/FairVoice",
                  test_percentage=0.2,
                  sample_equality=False,
                  sample_cap=default,
                  strong_equality=False,
                  sample_groups_equality=False,
                  tot_test_users=None,
                  test_equality=default,
                  test_per_lang=False,
                  load_test_users=False):
    if test_equality is default:
        test_equality = ["random"]

    if sample_cap is default:
        sample_cap = 100

    csvs = []
    last_info_name = ""
    if info_start_id and isinstance(metadata, str):
        last_info = [file for file in os.listdir(dir_metadata)
                     if file.startswith("info_metadata_{}_".format(info_start_id))][0]
        last_info_name = last_info
        info_end_id = int(os.path.splitext(last_info)[0].split('_')[-1])

        last_info = pd.read_csv(os.path.join(dir_metadata, last_info), sep=',', encoding=encoding)

        for file in os.listdir(dir_metadata):
            meta_id = re.search(r'(?<={})\d+'.format(os.path.splitext(metadata)[0]), file)
            if meta_id:
                if info_start_id <= int(meta_id[0]) <= info_end_id:
                    csvs.append(file)
    elif isinstance(metadata, list):
        csvs = metadata
        meta_id = re.search(r'(?<=.)\d+', csvs[0])
        for file in os.listdir(dir_metadata):
            if file.startswith("info_metadata_"):
                info_start_id, info_end_id = map(int, os.path.splitext(file)[0].split('_')[2:])
                if meta_id:
                    if info_start_id <= int(meta_id[0]) <= info_end_id:
                        last_info_name = file
                        last_info = pd.read_csv(os.path.join(dir_metadata, file), sep=',', encoding=encoding)
                        break
                else:
                    raise ValueError("bad format of {}".format(csvs[0]))
    else:
        raise ValueError("info_start_id expects metadata as str, or metadata should be a list of files")

    if not os.path.exists(os.path.join(dir_metadata, 'train')):
        os.mkdir(os.path.join(dir_metadata, 'train'))
    if not os.path.exists(os.path.join(dir_metadata, 'test')):
        os.mkdir(os.path.join(dir_metadata, 'test'))

    if not os.path.exists(os.path.join(dir_metadata, 'train', "{}_{}".format(info_start_id, info_end_id))):
        os.mkdir(os.path.join(dir_metadata, 'train', "{}_{}".format(info_start_id, info_end_id)))
    else:
        shutil.rmtree(os.path.join(dir_metadata, 'train', "{}_{}".format(info_start_id, info_end_id)))
        os.mkdir(os.path.join(dir_metadata, 'train', "{}_{}".format(info_start_id, info_end_id)))
    if not os.path.exists(os.path.join(dir_metadata, 'test', "{}_{}".format(info_start_id, info_end_id))):
        os.mkdir(os.path.join(dir_metadata, 'test', "{}_{}".format(info_start_id, info_end_id)))
    else:
        shutil.rmtree(os.path.join(dir_metadata, 'test', "{}_{}".format(info_start_id, info_end_id)))
        os.mkdir(os.path.join(dir_metadata, 'test', "{}_{}".format(info_start_id, info_end_id)))

    stats = {
        "filename": [],
        "train users": [],
        "train user samples": [],
        "train % F-M": [],
        "train % Y-O": [],
        "train languages": [],
        "test users": [],
        "test user samples": [],
        "comments": []
    }

    for meta_csv in csvs:
        stats["filename"].append(meta_csv)
        stats["comments"].append('test_percentage = {}\tsample_equality = {}\tsample_cap = {}\tstrong_equality = {}\t'
                                 'sample_groups_equality = {}\t tot_test_users = {}\t'.format(test_percentage,
                                                                                              sample_equality,
                                                                                              sample_cap,
                                                                                              strong_equality,
                                                                                              sample_groups_equality,
                                                                                              tot_test_users))
        train_users = []
        test_users = []
        labels = pd.read_csv(os.path.join(dir_metadata, meta_csv), sep=',', encoding=encoding)

        languages = labels.groupby("language_l1")

        csv_info = last_info[last_info["name"] == meta_csv]
        info_langs = csv_info["group by language"].to_numpy()
        info_genders = csv_info["female male ratio"].to_numpy()
        info_age = csv_info["young old ratio"].to_numpy()
        y_cap = csv_info["young cap"].to_numpy()[0]
        info_mode = csv_info["mode"].to_numpy()[0]

        if not isinstance(y_cap, float):
            ages = {k: "young" if ages_vals.index(y_cap) > i else "old" for i, k in enumerate(ages_vals)}

        if not isinstance(info_langs[0], float):
            local_mode = re.search(r'(?<=[(]).*(?=[)])', info_mode)[0]
            if local_mode == 'merge':
                if '[' in info_langs[0]:
                    lang_data = ast.literal_eval(info_langs[0])
                else:
                    lang_data = [info_langs[0]]
            else:
                lang_data = [info_langs[0].split('->')[1]]

            stats["train languages"].append(lang_data)

            for lang in lang_data:
                lang_df = languages.get_group(lang)
                train_users, test_users, valid = process_split_fm_yo_ratios(lang_df,
                                                                            info_genders,
                                                                            info_age,
                                                                            ages if "ages" in locals() else None,
                                                                            train_users,
                                                                            test_users,
                                                                            test_percentage,
                                                                            strong_equality,
                                                                            tot_test_users,
                                                                            load_test_users=load_test_users,
                                                                            lang=lang,
                                                                            dir_metadata=dir_metadata)
                # distinct case
                if not valid:
                    for _lang in lang_data:
                        tr_u, te_u = get_train_test(languages.get_group(_lang),
                                                    test_percentage=test_percentage,
                                                    strong_equality=strong_equality,
                                                    tot_test_users=tot_test_users,
                                                    load_test_users=load_test_users,
                                                    dir_metadata=dir_metadata,
                                                    lang=_lang)
                        train_users = train_users + tr_u
                        test_users = test_users + te_u
                    break
        else:
            stats["train languages"].append('N/A')
            train_users, test_users, _ = process_split_fm_yo_ratios(labels,
                                                                    info_genders,
                                                                    info_age,
                                                                    ages if "ages" in locals() else None,
                                                                    train_users,
                                                                    test_users,
                                                                    test_percentage,
                                                                    strong_equality,
                                                                    tot_test_users)

        data = {'audio': [], 'label': []}
        train_id = 0

        young_cap = None
        if 'y_cap' in locals():
            if not isinstance(y_cap, float):
                young_cap = y_cap

        meta_id = re.search(r'(?<=.)\d+', meta_csv)
        meta_id = int(meta_id[0])

        stats["train users"].append(len(train_users))

        train_df = pd.DataFrame()
        for t_lang, t_id in train_users:
            train_df = pd.concat([train_df, labels[(labels["language_l1"] == t_lang) & (labels["id_user"] == t_id)]])

        if not isinstance(info_genders[0], float):
            train_df_gender = train_df.groupby("gender")
            if len(train_df_gender.groups) == 2:
                stats["train % F-M"].append((round(len(train_df_gender.get_group("female"))/len(train_df), 2),
                                             round(len(train_df_gender.get_group("male"))/len(train_df), 2)))
            elif "female" in train_df_gender.groups:
                stats["train % F-M"].append((round(len(train_df_gender.get_group("female")) / len(train_df), 2), "N/A"))
            else:
                stats["train % F-M"].append(("N/A", round(len(train_df_gender.get_group("male")) / len(train_df), 2)))
        else:
            stats["train % F-M"].append("N/A")

        if young_cap:
            train_df_age = train_df.set_index("age").groupby(ages)
            if len(train_df_age.groups) == 2:
                stats["train % Y-O"].append((round(len(train_df_age.get_group("young")) / len(train_df), 2),
                                             round(len(train_df_age.get_group("old")) / len(train_df), 2)))
            elif "young" in train_df_age.groups:
                stats["train % Y-O"].append((round(len(train_df_age.get_group("young")) / len(train_df), 2), "N/A"))
            else:
                stats["train % Y-O"].append(("N/A", round(len(train_df_age.get_group("old")) / len(train_df), 2)))
        else:
            stats["train % Y-O"].append("N/A")

        tot_audios = 0
        for user in train_users:
            tot_audios += add_user_audios(data, user, labels, sample_cap, sample_equality, dir_dataset,
                                          extra_data=train_id)
            train_id += 1
        stats["train user samples"].append(tot_audios)

        meta_mode = re.search(r'.*(?= [(])', info_mode)[0]

        if not isinstance(info_langs[0], float):
            train_df.to_csv(
                os.path.join(dir_metadata, "train", "{}_{}".format(info_start_id, info_end_id),
                             "{}_metadata-{}-train.csv".format(meta_id, info_langs[0])),
                encoding='utf-8',
                index=False
            )

            pd.DataFrame(data=data).to_csv(
                os.path.join(dir_metadata, "train", "{}_{}".format(info_start_id, info_end_id),
                             "{}_{}-train.csv".format(meta_id, info_langs[0])),
                encoding='utf-8',
                index=False
            )

            if sample_groups_equality:
                eq_train = equalize_train_group_samples(os.path.join(dir_metadata, "train",
                                                                     "{}_{}".format(info_start_id, info_end_id),
                                                                     "{}_{}-train.csv".format(meta_id, info_langs[0])),
                                                        metadata=train_df,
                                                        young_cap=young_cap)

                eq_train.to_csv(
                    os.path.join(dir_metadata, "train", "{}_{}".format(info_start_id, info_end_id),
                                 "{}_{}-train.csv".format(meta_id, info_langs[0])),
                    encoding='utf-8',
                    index=False
                )

                train_lables = pd.read_csv(
                    os.path.join(dir_metadata, "train", "{}_{}".format(info_start_id, info_end_id),
                                 "{}_{}-train.csv".format(meta_id, info_langs[0])),
                    sep=',',
                    encoding='utf-8'
                )

                grouped_train_labels = train_lables.groupby("label")
                for gr_train in grouped_train_labels.groups:
                    group_train = grouped_train_labels.get_group(gr_train)
                    lang_train, id_user = os.path.split(os.path.split(group_train["audio"].to_numpy()[0])[0])
                    train_df.loc[(train_df["language_l1"] == lang_train) & (train_df["id_user"] == id_user),
                                 ["n_sample"]] = len(group_train)

                train_df.to_csv(
                    os.path.join(dir_metadata, "train", "{}_{}".format(info_start_id, info_end_id),
                                 "{}_metadata-{}-train.csv".format(meta_id, info_langs[0])),
                    encoding='utf-8',
                    index=False
                )

        else:
            pd.DataFrame(data=data).to_csv(
                os.path.join(dir_metadata, "train", "{}_{}".format(info_start_id, info_end_id),
                             "{}_train.csv".format(meta_id)),
                encoding='utf-8',
                index=False
            )

        stats["test users"].append(len(test_users))
        test_user_samples = []

        for idx_t_e, t_e in enumerate(test_equality):
            if test_per_lang:
                per_lang_list = languages.groups
            else:
                per_lang_list = [None]
            for lang in per_lang_list:
                _test_users = test_users
                data = {'audio_1': [], 'audio_2': [], 'age_1': [], 'age_2': [], 'gender_1': [], 'gender_2': [],
                        'label': []}
                if lang:
                    _test_users = [_user for _user in _test_users if _user[0] == lang]

                tot_audios = 0
                for user in _test_users:
                    age_data = None
                    if young_cap:
                        age_data = (ages, _test_users, t_e)
                    else:
                        age_data = (None, _test_users, t_e)
                    tot_audios += add_user_audios(data, user, labels, sample_cap, sample_equality, dir_dataset,
                                                  extra_data=age_data)
                out_test_sample_stat = '{}_{} {} {}'.format(idx_t_e + 1, t_e, lang, tot_audios) if lang \
                    else '{}_{} {}'.format(idx_t_e + 1, t_e, tot_audios)
                test_user_samples.append(out_test_sample_stat)

                out_test = "{}_{}-test{}.csv".format(meta_id, lang, idx_t_e + 1) if lang \
                    else "{}_test{}.csv".format(meta_id, idx_t_e + 1)

                pd.DataFrame(data=data).to_csv(
                    os.path.join(dir_metadata, "test", "{}_{}".format(info_start_id, info_end_id), out_test),
                    encoding='utf-8',
                    index=False
                )
        stats["test user samples"].append(test_user_samples)

        test_df = pd.DataFrame()
        for t_lang, t_id in test_users:
            test_df = pd.concat([test_df, labels[(labels["language_l1"] == t_lang) & (labels["id_user"] == t_id)]])

        if not isinstance(info_langs[0], float):
            test_df.to_csv(
                os.path.join(dir_metadata, "test", "{}_{}".format(info_start_id, info_end_id),
                             "{}_metadata-{}-test.csv".format(meta_id, info_langs[0])),
                encoding='utf-8',
                index=False
            )

    pd.DataFrame(data=stats).to_csv(
        os.path.join(dir_metadata,
                     "train_test_stats_{}_{}.csv".format(info_start_id, info_end_id)
                     ),
        encoding='utf-8',
        index=False
    )


def process_split_fm_yo_ratios(df,
                               info_genders,
                               info_age,
                               ages,
                               train_users,
                               test_users,
                               test_percentage,
                               strong_equality,
                               tot_test_users,
                               load_test_users=False,
                               lang=None,
                               dir_metadata=None):
    if not isinstance(info_genders[0], float):
        genders = df.groupby("gender")

        males = genders.get_group("male")
        females = genders.get_group("female")

        if not isinstance(info_age[0], float):
            male_by_age = males.set_index("age").groupby(ages)
            male_youngs = male_by_age.get_group("young").reset_index()
            male_olds = male_by_age.get_group("old").reset_index()

            female_by_age = females.set_index("age").groupby(ages)
            female_youngs = female_by_age.get_group("young").reset_index()
            female_olds = female_by_age.get_group("old").reset_index()

            tr_u, te_u = get_train_test(male_youngs, male_olds, female_youngs, female_olds,
                                        test_percentage=test_percentage,
                                        strong_equality=strong_equality,
                                        tot_test_users=tot_test_users,
                                        dir_metadata=dir_metadata,
                                        load_test_users=load_test_users,
                                        lang=lang)
            _train_users = train_users + tr_u
            _test_users = test_users + te_u
            return _train_users, _test_users, True
        else:
            tr_u, te_u = get_train_test(females, males,
                                        test_percentage=test_percentage,
                                        strong_equality=strong_equality,
                                        tot_test_users=tot_test_users,
                                        dir_metadata=dir_metadata,
                                        load_test_users=load_test_users,
                                        lang=lang)
            _train_users = train_users + tr_u
            _test_users = test_users + te_u
            return _train_users, _test_users, True

    elif not isinstance(info_age[0], float):
        df_by_age = df.set_index("age").groupby(ages)

        youngs = df_by_age.get_group("young").reset_index()
        olds = df_by_age.get_group("old").reset_index()

        tr_u, te_u = get_train_test(youngs, olds,
                                    test_percentage=test_percentage,
                                    strong_equality=strong_equality,
                                    tot_test_users=tot_test_users,
                                    dir_metadata=dir_metadata,
                                    load_test_users=load_test_users,
                                    lang=lang)
        _train_users = train_users + tr_u
        _test_users = test_users + te_u
        return _train_users, _test_users, True
    else:
        return train_users, test_users, False


def get_train_test(*args,
                   test_percentage=None,
                   strong_equality=False,
                   tot_test_users=None,
                   dir_metadata=None,
                   load_test_users=False,
                   lang=None):
    """
    :param tot_test_users: if not None it is equal to the number of users used for testing
    :param args: takes as many arguments as the number of groups created (4 groups for male_youngs, male_olds,
                 female_youngs, female_olds)
    :param test_percentage: percentage of the dataset to be used for testing
    :param strong_equality: for each group N users are taken for tests. N is chosen in this way:
                                        N * groups = len(dataset) * test_percentage
                            it can happen that some groups are represented by a number of users less than N.
                            In this situation:
                            - strong_equality = True ensure the equality of users in the groups for testing
                            changing the value of N = len(group with least number of users), but the number of users
                            for testing does not satisfy the test_percentage.
                            - strong_equality = False does not satisfy the equality of users among the groups, and
                            satisfies the test_percentage taking more users from the most represented group

    :return: two arrays of id_users, each id_user as str (train_users, test_users)
    """
    _train_users = []
    _test_users = []

    loaded_test_users = None
    if dir_metadata and load_test_users and lang:
        if tot_test_users:
            loaded_test_users = TestUsersLoader(
                os.path.join(dir_metadata, 'test_users_{}_{}'.format(lang, tot_test_users))
            )
        else:
            loaded_test_users = TestUsersLoader(
                os.path.join(dir_metadata, 'test_users_{}_{}'.format(lang, test_percentage))
            )

    groups_len = [len(df) for df in args]

    if not tot_test_users:
        tot_test_users = round(sum(groups_len) * test_percentage - ROUND_CORRECTOR)
    test_len = round(tot_test_users / len(args) - ROUND_CORRECTOR)

    min_df = min(groups_len)

    # not enough users to take "test_len" users from each group for testing
    if min_df < test_len:
        if strong_equality:
            test_len = min_df
        else:
            deficit = sum([test_len - df_len if df_len < test_len else 0 for df_len in groups_len])
            most_rep_df_len = max(groups_len) + deficit

    loaded_data = None
    if loaded_test_users:
        if len(loaded_test_users.test_users) > 0:
            loaded_data = loaded_test_users.test_users

    for df in args:
        lang_series = df["language_l1"].to_numpy(copy=True)
        id_series = df["id_user"].to_numpy(copy=True)
        _users = list(zip(lang_series, id_series))
        shuffle(_users)

        test_cut = test_len
        if "most_rep_df_len" in locals():
            if len(df) == most_rep_df_len:
                test_cut = most_rep_df_len

        users_for_testing = _users[:test_cut]
        users_for_training = _users[test_cut:]

        if loaded_data:
            users_for_testing = [_u for _u in _users if _u in loaded_data]
            users_for_training = [_u for _u in _users if _u not in loaded_data]
        elif loaded_test_users:
            loaded_test_users.test_users = loaded_test_users.test_users + users_for_testing

        _test_users = _test_users + users_for_testing
        _train_users = _train_users + users_for_training

    if loaded_test_users:
        loaded_test_users.save()

    return _train_users, _test_users


def add_user_audios(data, user, labels, sample_cap, sample_equality, dir_dataset, extra_data=None):
    TEST_MAX_AUDIOS = 16

    audios = [file for file in os.listdir(os.path.join(dir_dataset, user[0], user[1]))
              if os.path.splitext(file)[1] == ".wav" or os.path.splitext(file)[1] == ".mp3"]

    if sample_equality:
        shuffle(audios)
        if isinstance(sample_cap, int):
            audios = audios[:sample_cap]
        else:
            if sample_cap[user[0]] != 'max':
                audios = audios[:sample_cap[user[0]]]
        # elif len(audios) < min_sample:
        #    raise ValueError("Metadata contains users with n_sample < min_sample chosen as argument.\n"
        #                     "min_sample = {}".format(min_sample))

    # training
    if isinstance(extra_data, int):
        train_id = extra_data

        data['audio'].extend([os.path.join(user[0], user[1], audio) for audio in audios])
        data['label'].extend([train_id] * len(audios))

        n_audios_stat = len(audios)

    # testing
    else:
        ages, test_users, test_eq = extra_data
        user_gender = labels[(labels["language_l1"] == user[0]) & (labels["id_user"] == user[1])]["gender"].to_numpy()[0]
        user_age = labels[(labels["language_l1"] == user[0]) & (labels["id_user"] == user[1])]["age"].to_numpy()[0]
        set_audios = set()

        if not sample_equality:
            shuffle(audios)

        audios = audios[:min(len(audios), TEST_MAX_AUDIOS)]

        # adding to the csv the triplets of audios of same user (label: 1)
        for audio1 in audios[:len(audios) // 2]:
            for audio2 in audios[len(audios) // 2:]:
                set_audios.add((user[1], audio1))
                set_audios.add((user[1], audio2))
                data['audio_1'].append(os.path.join(user[0], user[1], audio1))
                data['audio_2'].append(os.path.join(user[0], user[1], audio2))
                if ages:
                    data['age_1'].append(ages[user_age])
                else:
                    data['age_1'].append('')
                data['age_2'].append('')
                data['gender_1'].append(user_gender)
                data['gender_2'].append('')
                data['label'].append(1)

        index_user = test_users.index(user)

        only_random_possible = True

        if not isinstance(user_gender, float):
            if test_eq == "gender":
                only_random_possible = False
                not_found = True
                while not_found:
                    other_user = random.randint(0, len(test_users) - 1)
                    if labels[(labels["language_l1"] == test_users[other_user][0]) &
                              (labels["id_user"] == test_users[other_user][1])]["gender"].to_numpy()[0] == user_gender \
                            and (index_user != other_user):
                        not_found = False

        if ages:
            if test_eq == "age":
                only_random_possible = False
                not_found = True
                while not_found:
                    other_user = random.randint(0, len(test_users) - 1)
                    if ages[labels[(labels["language_l1"] == test_users[other_user][0]) &
                            (labels["id_user"] == test_users[other_user][1])]["age"].to_numpy()[0]] == ages[user_age] \
                            and (index_user != other_user):
                        not_found = False

        if test_eq == "random" or only_random_possible:
            other_user = index_user
            while other_user == index_user:
                other_user = random.randint(0, len(test_users) - 1)

        other_user = test_users[other_user]
        other_gender = labels[(labels["language_l1"] == other_user[0]) &
                              (labels["id_user"] == other_user[1])]["gender"].to_numpy()[0]
        other_age = labels[(labels["language_l1"] == other_user[0]) &
                           (labels["id_user"] == other_user[1])]["age"].to_numpy()[0]

        other_audios = [file for file in os.listdir(os.path.join(dir_dataset, other_user[0], other_user[1]))
                        if os.path.splitext(file)[1] == ".wav" or os.path.splitext(file)[1] == ".mp3"]
        shuffle(other_audios)
        other_audios = other_audios[:min(len(other_audios), TEST_MAX_AUDIOS)]

        # adding to the csv the triplets of audios of different users (label: 0)
        for audio1 in audios[:min(len(audios), TEST_MAX_AUDIOS // 2)]:
            for audio2 in other_audios[:min(len(other_audios), TEST_MAX_AUDIOS // 2)]:
                set_audios.add((user[1], audio1))
                set_audios.add((other_user[1], audio2))
                data['audio_1'].append(os.path.join(user[0], user[1], audio1))
                data['audio_2'].append(os.path.join(other_user[0], other_user[1], audio2))
                if ages:
                    data['age_1'].append(ages[user_age])
                    data['age_2'].append(ages[other_age])
                else:
                    data['age_1'].append('')
                    data['age_2'].append('')
                data['gender_1'].append(user_gender)
                data['gender_2'].append(other_gender)
                data['label'].append(0)

        n_audios_stat = len(set_audios)

    return n_audios_stat


def equalize_train_group_samples(file, metadata=None, young_cap="fourties"):
    train_data = pd.read_csv(file, sep=',', encoding='utf-8')
    if isinstance(metadata, str):
        meta = pd.read_csv(metadata, sep=',', encoding='utf-8')
    else:
        meta = metadata

    train_df = pd.DataFrame()
    train_labels = train_data.groupby("label")
    users = {}

    # generate metadata_train from train csv
    for gr in train_labels.groups:
        group_df = train_labels.get_group(gr)
        split_lang_id = os.path.split(os.path.split(group_df["audio"].to_numpy()[0])[0])
        users['_'.join(split_lang_id)] = (gr, len(group_df))
        train_df = pd.concat([train_df,
                              meta[(meta["language_l1"] == split_lang_id[0]) & (meta["id_user"] == split_lang_id[1])]])

    if young_cap:
        ages = {k: "young" if ages_vals.index(young_cap) > i else "old" for i, k in enumerate(ages_vals)}

    groups_labels = []
    tots = [0]*4
    tot_index = 0
    gender_df = train_df.groupby("gender")
    for gender in gender_df.groups:
        gender_group = gender_df.get_group(gender)
        if young_cap:
            age_df = gender_group.set_index("age").groupby(ages)
            for age in age_df.groups:
                age_group = age_df.get_group(age)
                gr_labels = []
                for row in age_group.itertuples():
                    user = users['{}_{}'.format(row.language_l1, row.id_user)]
                    tots[tot_index] += user[1]
                    gr_labels.append(user[0])
                groups_labels.append(gr_labels)
                tot_index += 1
        else:
            raise NotImplementedError()

    remove_factor = 0.4
    min_to_not_remove = 40
    min_tot = min(tots)
    for i in range(4):
        i_labels = iter(groups_labels[i])
        while tots[i] != min_tot:
            try:
                label = next(i_labels)
            except StopIteration:
                i_labels = iter(groups_labels[i])
                label = next(i_labels)
            if len(train_labels.get_group(label)) > min_to_not_remove:
                remove_n_rows = min((tots[i] - min_tot), round(len(train_labels.get_group(label)) * remove_factor))
            else:
                remove_n_rows = 0
            new_rows = len(train_labels.get_group(label)) - remove_n_rows
            train_data = pd.concat([train_labels.get_group(label).sample(new_rows),
                                    train_data[train_data["label"] != label]])
            train_labels = train_data.groupby("label")
            tots[i] -= remove_n_rows
    print("equalize_groups: {}".format(tots))

    return train_data


if __name__ == "__main__":
    _info_start_id = prepare_common_voice(group_by_lang=["English", "Spanish"],
                                          female_male_ratio=[(0.5, 0.5)],
                                          young_old_ratio=[(0.5, 0.5)],
                                          lang_equality=True,
                                          mode=Modes.MERGE,
                                          tot_test_users=100)
    split_dataset(info_start_id=_info_start_id,
                  sample_equality=True,
                  strong_equality=True,
                  sample_groups_equality=True,
                  tot_test_users=100,
                  sample_cap={"English": 221, "Spanish": 'max'},
                  test_equality=["age", "gender", "random"],
                  test_per_lang=True,
                  load_test_users=True)
