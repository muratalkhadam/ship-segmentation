import numpy as np
import random


def split_ids(df, number, with_ships=0.75, train_size=0.8):
    with_ships_num_train = int(number * with_ships * train_size)
    with_ships_df = list(df[df['EncodedPixels'] != '']['ImageId'].values)
    with_ships_df = random.sample(with_ships_df, with_ships_num_train)

    without_ships_num_train = int(number * train_size - with_ships_num_train)
    without_ships_df = list(df[df['EncodedPixels'] == '']['ImageId'].values)
    without_ships_df = random.sample(without_ships_df, without_ships_num_train)

    train = np.concatenate((with_ships_df, without_ships_df))
    np.random.shuffle(train)

    with_ships_num_test = int((number - len(train)) * with_ships)
    with_ships_df = list(df[(df['EncodedPixels'] != '') & (~df['ImageId'].isin(train))]['ImageId'].values)
    with_ships_df = random.sample(with_ships_df, with_ships_num_test)

    without_ships_num_test = number - len(train) - with_ships_num_test
    without_ships_df = list(df[(df['EncodedPixels'] == '') & (~df['ImageId'].isin(train))]['ImageId'].values)
    without_ships_df = random.sample(without_ships_df, without_ships_num_test)

    test = np.concatenate((with_ships_df, without_ships_df))
    np.random.shuffle(test)

    return train, test
