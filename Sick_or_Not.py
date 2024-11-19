import os
import tensorflow as tf

dire = os.getcwd()
train_dir = dire+"\\images_512\\train"
val_dir = dire+"\\images_512\\valid"
class_name = ["Sick", "Healthy"]
CONFIG = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
}

train = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    class_names=class_name,
    color_mode='rgb',
    batch_size=CONFIG["BATCH_SIZE"],
    image_size=(CONFIG["IM_SIZE"], CONFIG["IM_SIZE"]),
    shuffle=True,
    seed=42)
val = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    class_names=class_name,
    color_mode='rgb',
    batch_size=CONFIG["BATCH_SIZE"],
    image_size=(CONFIG["IM_SIZE"], CONFIG["IM_SIZE"]),
    shuffle=True,
    seed=42)


# final data (augmented)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Resizing(512, 512))
model.add(tf.keras.layers.Rescaling(1. / 511))
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(512, 512, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train,
                    batch_size=32,
                    epochs=5,
                    validation_data=val,
                    validation_batch_size=16
                    )


model.save("Sick_or_Not_1.4.keras")
