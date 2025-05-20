import tensorflow as tf
from tensorflow.keras import layers, models, losses

def create_cnn(height: int, width: int) -> tf.keras.Model:
    input_layer = layers.Input(shape=(height, width, 11))
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(input_layer)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Conv2D(1, 1, padding='same')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
    return model
