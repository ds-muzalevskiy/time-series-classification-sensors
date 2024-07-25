from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical

def sampling (X, y):
    #sampling is doing for ndim<=2 so doing reshaping from ndim=3 to ndim=2
    train_X_TwoDim = X.reshape(X.shape[0],-1)

    #using oversampling for fit_sample train dataset    
    sm = RandomOverSampler(random_state=42, sampling_strategy='auto')
    train_X, train_y = sm.fit_sample(train_X_TwoDim, y)
    
    #reshape features back to ndim=3 and switch target to_categorical for using in model
    train_X = train_X.reshape(train_X.shape[0], X.shape[1],X.shape[2])
    
    if train_y.shape[1] == 1:
        train_y = to_categorical(train_y)
    
    return train_X, train_y