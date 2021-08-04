import tensorflow as tf


def retrain_best_model(best_model, x_train, x_test, config):
    batch_size = config['batch_size']
    epochs = config['retrain_epochs']
    should_early_stop = config['should_early_stop']

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = best_model.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, x_test),
        shuffle=False,
        callbacks=[stop_early] if should_early_stop else [],
    ).history

    train_preds = best_model.predict(x_train, batch_size=batch_size)
    test_preds = best_model.predict(x_test, batch_size=batch_size)

    return history, train_preds, test_preds, best_model
