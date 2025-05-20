import os
from tensorflow.keras.callbacks import EarlyStopping
from configs import *
from models.model import create_cnn
from utils.dataset import generate_training_data

def main():
    train_inputs, train_outputs = generate_training_data(WIDTH, HEIGHT, NUM_MINES, NUM_TRAIN_EXAMPLES)
    test_inputs, test_outputs = generate_training_data(WIDTH, HEIGHT, NUM_MINES, NUM_TEST_EXAMPLES)

    model = create_cnn(HEIGHT, WIDTH)

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    
    model.fit(
        train_inputs, train_outputs,
        validation_data=(test_inputs, test_outputs),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
