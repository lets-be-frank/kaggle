import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close('all')
# %% MNIST dataset
data = pd.read_csv('train.csv')
y = data.values[:,0]
X = data.values[:,1:]
# plot 25 random digits
plt.figure()
for j, i in enumerate(np.random.choice(np.arange(len(y)), 25)): 
    print('Plotting %i of 25'%(j+1))
    plt.subplot(5,5, j+1)
    grayscale = X[i,:]
    a = grayscale.reshape(28,28)
    # plot image
    plt.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(y[i],fontweight='bold')

plt.tight_layout()

# %% Tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras import activations
# from tensorflow.keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X/255, y, test_size=0.25, random_state=123)


# %% Constructing as matrices
Xi_train = []
Xi_test = []
for i in range(len(y_train)):
    Xi_train.append(X_train[i,:].reshape(28,28,1))
for i in range(len(y_test)):
    Xi_test.append(X_test[i,:].reshape(28,28,1))

Xi_train = np.array(Xi_train)
Xi_test = np.array(Xi_test)

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.reshape(-1,1))
y_train = y_train.toarray()
y_test = enc.transform(y_test.reshape(-1,1)).toarray()

# plot 25 random digits
plt.figure()
for j, i in enumerate(np.random.choice(np.arange(len(y_train)), 25)): 
    print('Plotting %i of 25'%(j+1))
    plt.subplot(5,5, j+1)
    # plot image
    plt.imshow(Xi_train[i,:,:,0], cmap='viridis', vmin=0, vmax=1)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(np.argmax(y_train,axis=1)[i],fontweight='bold')

plt.tight_layout()

# %% Constructing CNN
print('Constructing CNN')
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# %% adding the Dense Model afterwards
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# model.add(activations('softmax'))

# %% Compiling Model
print('Compiling Model')
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# %% Training Model
model.fit(Xi_train, y_train, epochs=5)

# %% plot sample predictions
# plot 25 random digits

plt.figure()
for j, i in enumerate(np.random.choice(np.arange(len(y_test)), 25)): 
    print('Predicting %i of 25'%(j+1))
    plt.subplot(5,5, j+1)
    # plot image
    plt.imshow(Xi_test[i,:,:,0], cmap='viridis', vmin=0, vmax=1)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    pred = model.predict(Xi_test[i,:,:,:].reshape(1,28,28,1))
    caption = 'Prediction: %i'%np.argmax(pred)
    
    plt.title(caption,fontweight='bold')

plt.tight_layout()
# %% Submission
submission_data = pd.read_csv('test.csv').values/255

Xs = []
for i in range(len(submission_data)):
    Xs.append(submission_data[i,:].reshape(28,28,1))
Xs = np.array(Xs)

pred_s = model.predict(Xs)
results = np.argmax(pred_s,axis=1)

df_output = pd.DataFrame({
    'ImageId':np.arange(len(submission_data))+1,
    'Label':results})
df_output.to_csv('submissionv1.csv',index=None)
