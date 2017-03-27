import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras import callbacks


K.set_image_dim_ordering('tf')
#martins variabler
train_dir = '/home/jason/tf_train/train_3/train'
val_dir = '/home/jason/tf_train/train_3/validate'
img_size = 299
gen_batch = 64
classes = 9
weights_path = '/home/jason/train_3'
epoker = 50
sdg = SGD(lr=0.0001, momentum=0.9)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.5)
samples = 20
log = '/home/jason/tf_train_train_3'
save_path = '/home/jason/tf_train/train_3/save'
tensorboard = callbacks.TensorBoard(log_dir=log, histogram_freq=1, write_graph=False)
drop = 0.5




base_model = InceptionV3(weights='imagenet', include_top=False)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

x = Dropout(drop)(x)
predictions = Dense(classes, activation='softmax')(x)


model = Model(input=base_model.input, output=predictions)


for layer in base_model.layers:
    layer.trainable = False

train_data = image.ImageDataGenerator(
                            rescale=1. / 255,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest',
                        )

test_datagen = image.ImageDataGenerator(
                            rescale=1. / 255,

                        )

validation_generator = test_datagen.flow_from_directory(
                            val_dir,
                            target_size=(img_size, img_size),
                            batch_size=gen_batch,
                            class_mode='categorical',
                            classes=['Agaricus_bisporus', 'Amanita_muscaria', 'Amanita_virosa', 'Boletus_edulis', 'Cantharellus_cibarius', 'Cortinarius_rubellus', 'Craterellus_cornucopioides', 'Lactarius_deterrimus', 'Suillus_variegatus'],
                            shuffle='False'
                        )

train_generator = train_data.flow_from_directory(
                            train_dir,
                            target_size=(img_size, img_size),
                            batch_size=gen_batch,
                            shuffle='False',
                            class_mode='categorical',
                            classes=['Agaricus_bisporus', 'Amanita_muscaria', 'Amanita_virosa', 'Boletus_edulis', 'Cantharellus_cibarius', 'Cortinarius_rubellus', 'Craterellus_cornucopioides', 'Lactarius_deterrimus', 'Suillus_variegatus'],
                        )

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
model.fit_generator(
                            train_generator,
                            samples_per_epoch=128 * samples,
                            nb_epoch=epoker,
                            validation_data=validation_generator,
                            nb_val_samples=32,
                            callbacks=[tensorboard]
                        )

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)


for layer in model.layers[:171]:
    layer.trainable = False
for layer in model.layers[171:]:
    layer.trainable = True


model.compile(optimizer=sdg, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
                            train_generator,
                            samples_per_epoch=128 * samples,
                            nb_epoch=epoker,
                            validation_data=validation_generator,
                            nb_val_samples=32
                        )
model.save(save_path)

test_datagen2 = image.ImageDataGenerator(rescale=1. / 255, dim_ordering=K.image_dim_ordering())

prediction_generator = test_datagen2.flow_from_directory(
                                val_dir, target_size=(img_size, img_size),
                                batch_size=gen_batch,
                                classes=['Agaricus_bisporus', 'Amanita_muscaria', 'Amanita_virosa', 'Boletus_edulis', 'Cantharellus_cibarius',
                                            'Cortinarius_rubellus', 'Craterellus_cornucopioides', 'Lactarius_deterrimus', 'Suillus_variegatus'],
                                shuffle='False'
                        )
model.predict_generator(prediction_generator)
