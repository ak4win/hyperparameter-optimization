import tensorflow as tf


def train_best_model(model_category, x_train, x_test, batch_size, epochs):
    model = tf.keras.models.load_model(
        f"/home/paperspace/hyperparameter-optimization/save_models/{model_category}"
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        x_train,
        x_train,
        batch_size,
        epochs=epochs,
        validation_data=(x_test, x_test),
        shuffle=False,
        callbacks=stop_early,
    ).history

    train_preds = model.predict(x_train, batch_size=batch_size)
    test_preds = model.predict(x_test, batch_size=batch_size)

    return history, train_preds, test_preds
