The implementation was based on the original source code (https://github.com/vlawhern/arl-eegmodels).

For the details of the implementation, please see our paper: https://www-sciencedirect-com.ornl.idm.oclc.org/science/article/pii/S0957417423008503.

# MTL-EEGModels
This is the implementation of MTL-CNNs for EEG signal classification with additional label information, using Keras and Tensorflow. 

## Requirements
- Python >= 3.7
- tensorflow == 2.X (both for CPU and GPU)
- numpy >= 1.0

## Usage Example
Import libraries and models
```python
from MTL_EEGModels import MTL_EEGNet, MTL_ShallowConvNet, MTL_DeepConvNet
```

Generate dummy data
```python
n_eeg_samples = 180 # number of EEG samples for single subject

n_channels = 62 # number of EEG channels used

n_times = 512 # time points based on sampling rate for EEG signal

x = np.random.rand(n_eeg_samples, n_channels, n_times)

y_emotion = np.random.randint(3, size=(n_eeg_samples, )) # three classes for emotion classification task (fear, sad, neutral)

y_context = np.random.randint(2, size=(n_eeg_samples, )) # two classes for context classification task (social, non-social)
```

Data reshaping
```python
x = np.expand_dims(x, axis = 3)

y_emotion = np_utils.to_categorical(y_emotion, num_classes=emotion_nb_classes)

y_context = np_utils.to_categorical(y_context, num_classes=context_nb_classes)
```

Parameter settings
```python
emotion_nb_classes = 3
context_nb_classes = 2
channels, samples = x_train.shape[1], x_train.shape[2]
gamma = 0.5 # weight proportion of tasks
n_epochs = 10
batch_size = 16
```

Define model
```python
model = MTL_EEGNet(emotion_nb_classes=emotion_nb_classes, context_nb_classes=context_nb_classes, Chans=channels, Samples=samples)
# model = MTL_ShallowConvNet(emotion_nb_classes=emotion_nb_classes, context_nb_classes=context_nb_classes, Chans=channels, Samples=samples)
# model = MTL_DeepConvNet(emotion_nb_classes=emotion_nb_classes, context_nb_classes=context_nb_classes, Chans=channels, Samples=samples)
```

Compile the model
```python
model.compile(optimizer='adam',
              loss={'emotion_output': 'categorical_crossentropy',
                    'context_output': 'categorical_crossentropy'},
                loss_weights={'emotion_output': gamma,
                              'context_output': 1-gamma},
                metrics=['accuracy'])
model.fit({'input': x_train},
          {'emotion_output': y_train_emotion, 'context_output': y_train_context},
           epochs=n_epoch, batch_size=batch_size, verbose=0)
```


