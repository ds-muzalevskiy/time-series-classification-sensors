import numpy as np
from mcfly import modelgen
from sklearn import neighbors, metrics as sklearnmetrics
import warnings
import json
import os
from keras.callbacks import EarlyStopping
from keras import metrics
from lib.balance_sample_weights import get_sample_weights


def train_models_on_samples(X_train, y_train, X_val, y_val, models,
                            nr_epochs=5, subset_size=100, verbose=True, outputfile=None,
                            model_path=None, early_stopping=True,class_weight=None,
                            batch_size=20, metric='accuracy'):

    X_train_sub = X_train[:subset_size, :, :]
    y_train_sub = y_train[:subset_size, :]

    metric_name = get_metric_name(metric)

    histories = []
    val_metrics = []
    val_losses = []
    for i, (model, params, model_types) in enumerate(models):
        if verbose:
            print('Training model %d' % i, model_types)
        model_metrics = [get_metric_name(name) for name in model.metrics]
        if metric_name not in model_metrics:
            raise ValueError(
                'Invalid metric. The model was not compiled with {} as metric'.format(metric_name))
        if early_stopping:
            callbacks = [
                EarlyStopping(monitor='val_acc', patience=0, verbose=verbose, mode='auto', restore_best_weights=True)]
        else:
            callbacks = []

        if class_weight:
            class_weight = get_sample_weights(y_train_sub)

        history = model.fit(X_train_sub, y_train_sub,
                            epochs=nr_epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            verbose=verbose,
                            callbacks=callbacks,
                            class_weight=class_weight)
        histories.append(history)

        val_metrics.append(history.history['val_' + metric_name][-1])
        val_losses.append(history.history['val_loss'][-1])
        if outputfile is not None:
            store_train_hist_as_json(params, model_types,
                                     history.history, outputfile)
        if model_path is not None:
                model.save(os.path.join(model_path, 'model_{}.h5'.format(i)))

    return histories, val_metrics, val_losses


def store_train_hist_as_json(params, model_type, history, outputfile, metric_name='acc'):

    jsondata = params.copy()
    for k in jsondata.keys():
        if isinstance(jsondata[k], np.ndarray):
            jsondata[k] = jsondata[k].tolist()
    jsondata['train_metric'] = history[metric_name]
    jsondata['train_loss'] = history['loss']
    jsondata['val_metric'] = history['val_' + metric_name]
    jsondata['val_loss'] = history['val_loss']
    jsondata['modeltype'] = model_type
    jsondata['metric'] = metric_name
    if os.path.isfile(outputfile):
        with open(outputfile, 'r') as outfile:
            previousdata = json.load(outfile)
    else:
        previousdata = []
    previousdata.append(jsondata)
    with open(outputfile, 'w') as outfile:
        json.dump(previousdata, outfile, sort_keys=True,
                  indent=4, ensure_ascii=False)


def find_best_architecture(X_train, y_train, X_val, y_val, verbose=True,
                           number_of_models=5, nr_epochs=5, subset_size=100,
                           outputpath=None, model_path=None, metric='accuracy',
                           **kwargs):

    models = modelgen.generate_models(X_train.shape, y_train.shape[1],
                                      number_of_models=number_of_models,
                                      metrics=[metric],
                                      **kwargs)
    histories, val_accuracies, val_losses = train_models_on_samples(X_train,
                                                                    y_train,
                                                                    X_val,
                                                                    y_val,
                                                                    models,
                                                                    nr_epochs,
                                                                    subset_size=subset_size,
                                                                    verbose=verbose,
                                                                    outputfile=outputpath,
                                                                    model_path=model_path,
                                                                    metric=metric)
    best_model_index = np.argmax(val_accuracies)
    best_model, best_params, best_model_type = models[best_model_index]
    
    if verbose:
        print('Best model: model ', best_model_index)
        print('Model type: ', best_model_type)
        print('Hyperparameters: ', best_params)
        print(str(metric) + ' on validation set: ',
              val_accuracies[best_model_index])

    return best_model, best_params, best_model_type


def get_metric_name(name):

    if name == 'acc' or name == 'accuracy':
        return 'acc'
    try:
        metric_fn = metrics.get(name)
        return metric_fn.__name__
    except:
        pass
    return name

