from numpy.random import seed; seed(1)

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Lambda,
    Flatten,
    Reshape,
    Input,
    Conv2D,
)

from lib.mse_without_nans import mean_squared_error_without_nans
from global_utils.layers.custom_conv2d_transpose import CustomConv2DTranspose
from global_utils.layers.max_pooling_with_argmax import MaxPoolWithArgMax
from global_utils.layers.unpooling_with_argmax import UnMaxPoolWithArgmax
from global_utils.layers.sampling import sample_from_latent_space

from global_utils.get_data_multi_note import read_and_preprocess_data
from global_utils.evaluation import smooth_output


def create_model(config: dict, batch_size=32, sequence_length=120) -> Model:
    inputs = Input(shape=(sequence_length, 1, 1), batch_size=batch_size)

    channels, kernel = 24, 12
    out_conv1 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(inputs)
    out_reshape1 = Reshape((-1, 1, 1))(out_conv1)
    out_pool1, mask1 = MaxPoolWithArgMax()(out_reshape1)

    channels, kernel = 12, 9
    out_conv2 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_pool1)
    out_reshape2 = Reshape((-1, 1, 1))(out_conv2)
    out_pool2, mask2 = MaxPoolWithArgMax()(out_reshape2)

    out_max_pool_1, mask3 = MaxPoolWithArgMax()(out_pool2)

    channels, kernel = 12, 5
    out_conv3 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_max_pool_1)
    out_reshape3 = Reshape((-1, 1, 1))(out_conv3)
    out_pool3, mask4 = MaxPoolWithArgMax()(out_reshape3)

    channels, kernel = 12, 3
    out_conv4 = Conv2D(
        channels,
        (kernel, 1),
        strides=kernel,
        activation=config["encoder_activation"],
        padding="same",
    )(out_pool3)
    out_reshape4 = Reshape((-1, 1, 1))(out_conv4)
    out_pool4, mask5 = MaxPoolWithArgMax()(out_reshape4)

    out_max_pool_2, mask6 = MaxPoolWithArgMax()(out_pool4)

    out_flatten = Flatten()(out_max_pool_2)
    out_dense1 = Dense(
        config["dense_nodes"], activation=config["bottleneck_activation"]
    )(out_flatten)
    z_mean = Dense(5, activation=config["bottleneck_activation"], name="z_mean")(
        out_dense1
    )
    z_log_var = Dense(5, activation=config["bottleneck_activation"], name="z_log_var")(
        out_dense1
    )
    z = Lambda(sample_from_latent_space, output_shape=(5,), name="z")(
        [z_mean, z_log_var]
    )

    # Decoder
    de_out_dense2 = Dense(
        config["dense_nodes"], activation=config["bottleneck_activation"]
    )(z)
    de_out_dense3 = Dense(54, activation=config["bottleneck_activation"])(de_out_dense2)
    de_inverse_flatten = Reshape((-1, 1, 1))(de_out_dense3)

    de_out_pool5 = UnMaxPoolWithArgmax(stride=2)(de_inverse_flatten, mask6)

    channels, kernel = 12, 3
    de_out_pool6 = UnMaxPoolWithArgmax(stride=2)(de_out_pool5, mask5)
    de_out_reshape5 = Reshape(out_conv4.shape[1:])(de_out_pool6)
    de_out_transcov1 = CustomConv2DTranspose(
        channels, kernel, out_pool3.shape, activation=config["decoder_activation"]
    )(de_out_reshape5)

    channels, kernel = 12, 5
    de_out_pool7 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov1, mask4)
    de_out_reshape6 = Reshape(out_conv3.shape[1:])(de_out_pool7)
    de_out_transcov2 = CustomConv2DTranspose(
        channels,
        kernel,
        out_max_pool_1.shape,
        activation=config["decoder_activation"],
    )(de_out_reshape6)

    de_out_pool8 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov2, mask3)

    channels, kernel = 12, 9
    de_out_pool9 = UnMaxPoolWithArgmax(stride=2)(de_out_pool8, mask2)
    de_out_reshape7 = Reshape(out_conv2.shape[1:])(de_out_pool9)
    de_out_transcov3 = CustomConv2DTranspose(
        channels, kernel, out_pool1.shape, activation=config["decoder_activation"]
    )(de_out_reshape7)

    channels, kernel = 24, 12
    de_out_pool10 = UnMaxPoolWithArgmax(stride=2)(de_out_transcov3, mask1)
    de_out_reshape8 = Reshape(out_conv1.shape[1:])(de_out_pool10)
    de_out_transcov4 = CustomConv2DTranspose(
        channels, kernel, inputs.shape, activation=config["decoder_activation"]
    )(de_out_reshape8)
    outputs = de_out_transcov4

    model = Model(inputs, mask1)

    if config["optimizer"] == "Adam":
        optimizer = keras.optimizers.Adam(lr=config["lr"])
    else:
        optimizer = keras.optimizers.SGD(
            lr=config["lr"], momentum=config["sgd_momentum"]
        )

    model.compile(
        loss=mean_squared_error_without_nans,
        optimizer=optimizer,
        metrics=[],
    )
    return model


if __name__ == "__main__":
    x_train, x_test = read_and_preprocess_data(
        sequence_length=120,
        batch_size=32,
        motes_train=[7],
        motes_test=[7],
    )

    # split train and test data
    train_test_cutoff = 320
    x_train = x_train[:train_test_cutoff, :, :]
    x_test = x_test[train_test_cutoff:, :, :]

    model = create_model({'encoder_activation': 'tanh', 'decoder_activation': 'tanh', 'dense_nodes': 33,
                          'bottleneck_activation': 'relu', 'lr': 0.001521356711612709, 'optimizer': 'Adam',
                          'sgd_momentum': 0.5091287212784572})
    history = model.fit(
        x_train,
        x_train,
        batch_size=32,
        epochs=500,
        validation_data=(x_test, x_test),
        shuffle=False,
        callbacks=[],
    ).history

    train_preds = model.predict(x_train, batch_size=32)
    test_preds = model.predict(x_test, batch_size=32)

    reconstruction, prms_diff = smooth_output(x_test, test_preds, smoothing_window=5)

    print(f"The percentual-RMS-difference for the configuration after the re-training is {prms_diff}")
