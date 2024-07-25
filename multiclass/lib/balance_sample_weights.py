import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from one_hot_to_categorical import one_hot_score_to_categorical


def generate_sample_weights(training_data, class_weight_dictionary):
    sample_weights = [class_weight_dictionary[np.where(one_hot_row==1)[0][0]] for one_hot_row in training_data]
    return np.asarray(sample_weights)


def get_class_weight_dictionary(y_one_hot):

    categorical_y = one_hot_score_to_categorical(y_one_hot)
    categorical_classes = np.unique(categorical_y)
    class_weights = compute_class_weight(class_weight='balanced', classes=categorical_classes, y=categorical_y)
    class_weight_dict = {a_class: a_weight for a_class, a_weight in zip(categorical_classes, class_weights)}
    return class_weight_dict


def get_sample_weights(y_one_hot):

    class_weights = get_class_weight_dictionary(y_one_hot)
    return generate_sample_weights(y_one_hot, class_weights)
