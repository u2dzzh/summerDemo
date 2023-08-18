import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from tools.recsysNN_utils import *
pd.set_option("display.precision", 1)


if __name__ == '__main__':
    # Load Data, set configuration variables
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
    num_item_features = item_train.shape[1] - 1  # remove movie id at train time
    uvs = 3  # user genre vector start
    ivs = 3  # item genre vector start
    u_s = 3  # start of columns to use in training, user
    i_s = 1  # start of columns to use in training, items
    scaledata = True  # applies the standard scalar to data if true
    print(f"Number of training vectors: {len(item_train)}")
    # scale training data
    if scaledata:
        item_train_save = item_train
        user_train_save = user_train

        scalerItem = StandardScaler()
        scalerItem.fit(item_train)
        item_train = scalerItem.transform(item_train)

        scalerUser = StandardScaler()
        scalerUser.fit(user_train)
        user_train = scalerUser.transform(user_train)

        print(np.allclose(item_train_save, scalerItem.inverse_transform(item_train)))
        print(np.allclose(user_train_save, scalerUser.inverse_transform(user_train)))
        # Split the Dataset
        item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
        user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
        y_train, y_test = train_test_split(y_train, train_size=0.80, shuffle=True, random_state=1)

        scaler = MinMaxScaler((-1, 1))
        scaler.fit(y_train.reshape(-1, 1))
        ynorm_train = scaler.transform(y_train.reshape(-1, 1))
        ynorm_test = scaler.transform(y_test.reshape(-1, 1))
        print(ynorm_train.shape, ynorm_test.shape)


        # Construct the Neural Network
        # GRADED_CELL
        # UNQ_C1

        num_outputs = 32
        tf.random.set_seed(1)
        user_NN = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_outputs)
        ])

        item_NN = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_outputs)
        ])

        # create the user input and point to the base network
        input_user = tf.keras.layers.Input(shape=(num_user_features))
        vu = user_NN(input_user)
        vu = tf.linalg.l2_normalize(vu, axis=1)

        # create the item input and point to the base network
        input_item = tf.keras.layers.Input(shape=(num_item_features))
        vm = item_NN(input_item)
        vm = tf.linalg.l2_normalize(vm, axis=1)

        # compute the dot product of the two vectors vu and vm
        output = tf.keras.layers.Dot(axes=1)([vu, vm])

        # specify the inputs and output of the model
        model = Model([input_user, input_item], output)

        model.summary()

        tf.random.set_seed(1)
        cost_fn = tf.keras.losses.MeanSquaredError()
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt,
                      loss=cost_fn)

        # Train
        tf.random.set_seed(1)
        model.fit([user_train[:, u_s:], item_train[:, i_s:]], ynorm_train, epochs=30)
        # Evaluation
        model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], ynorm_test)

        # Predictions for a New User
        new_user_id = 5000
        new_rating_ave = 1.0
        new_action = 1.0
        new_adventure = 1
        new_animation = 1
        new_childrens = 1
        new_comedy = 5
        new_crime = 1
        new_documentary = 1
        new_drama = 1
        new_fantasy = 1
        new_horror = 1
        new_mystery = 1
        new_romance = 5
        new_scifi = 5
        new_thriller = 1
        new_rating_count = 3

        user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                              new_action, new_adventure, new_animation, new_childrens,
                              new_comedy, new_crime, new_documentary,
                              new_drama, new_fantasy, new_horror, new_mystery,
                              new_romance, new_scifi, new_thriller]])

        # generate and replicate the user vector to match the number movies in the data set.
        user_vecs = gen_user_vecs(user_vec, len(item_vecs))

        # scale the vectors and make predictions for all movies. Return results sorted by rating.
        sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs, item_vecs, model, u_s, i_s,
                                                                              scaler, scalerUser, scalerItem,
                                                                              scaledata=scaledata)

        print_pred_movies(sorted_ypu, sorted_user, sorted_items, movie_dict, maxcount=10)

        # predict for an exist user
        uid = 36
        # form a set of user vectors. This is the same vector, transformed and repeated.
        user_vecs, y_vecs = get_user_vecs(uid, scalerUser.inverse_transform(user_train), item_vecs, user_to_genre)

        # scale the vectors and make predictions for all movies. Return results sorted by rating.
        sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs, item_vecs, model, u_s, i_s,
                                                                              scaler,
                                                                              scalerUser, scalerItem,
                                                                              scaledata=scaledata)
        sorted_y = y_vecs[sorted_index]

        # print sorted predictions
        print_existing_user(sorted_ypu, sorted_y.reshape(-1, 1), sorted_user, sorted_items, item_features, ivs, uvs,
                            movie_dict, maxcount=10)