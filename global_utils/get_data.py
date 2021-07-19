# ========================================================================================================================
#
# ========================================================================================================================
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_data(
    config,
    train_range=(0, 35000),
    test_range=(35000, 40000),
    cut_off_min=5,
    cut_off_max=45,
    should_scale=True,
    data_path="/work/data/IntelBerkeleyResearchLab.txt",
):
    """
    Load the temperature sensor data of the "Intel Berkeley Research Lab" dataset, clean it and scale it down.

    :parameters:
    config(int)           -- value that indicates which of the three possible configs (number of motes) to use
    train_range(tuple)    -- start and end of interval to cut out of the data for training
    test_range(tuple)     -- start and end of interval to cut out of the data for testing
    cut_off_min(number)   -- threshhold to discard all temperatures below that point
    cut_off_max(number)   -- threshhold to discard all temperatures above that point
    should_scale(boolean) -- switch between min-max-scaling data or not
    data_path(string)     -- path to the file containing all data

    :returns:
    x_train -- numpy array of shape dictated by config and train_range
    x_test  -- numpy array of shape dictated by config and test_range
    config  -- chosen config in case it needs to be reused later on
    """

    # Get relevant motes
    assert config in [1, 2, 3], "Invalid Input: config has to be one of 1,2 or 3"
    motes_config1 = [2]
    motes_config2 = motes_config1 + [6, 7, 8, 9, 10]
    motes_config3 = motes_config2 + [31, 35, 36]
    mote_ids = [motes_config1, motes_config2, motes_config3][config - 1]

    # Load, clean and preprocess data
    df = pd.read_csv(
        data_path,
        sep=" ",
        lineterminator="\n",
        names=[
            "date",
            "time",
            "epoch",
            "moteid",
            "temperature",
            "humidity",
            "light",
            "voltage",
        ],
    )

    # Clean outliers
    df.drop(
        df[(df["temperature"] < cut_off_min) | (df["temperature"] > cut_off_max)].index,
        inplace=True,
    )

    lower_bound = df["temperature"].mean() - 3 * df["temperature"].std()
    upper_bound = df["temperature"].mean() + 3 * df["temperature"].std()

    df.drop(
        df[(df["temperature"] < lower_bound) | (df["temperature"] > upper_bound)].index,
        inplace=True,
    )

    # Concatenate all relevant motes into one dataframe
    tmp_frames = []
    for mote_id in mote_ids:
        tmp_frame = df.loc[df["moteid"] == mote_id]["temperature"]
        tmp_frame = tmp_frame.reset_index(drop=True)
        tmp_frames.append(tmp_frame)
    df_full = pd.concat(tmp_frames, axis=1)

    # Get relevant intervals
    x_train_start = train_range[0]
    x_train_end = train_range[1]
    x_test_start = test_range[0]
    x_test_end = test_range[1]

    x_train = df_full[x_train_start:x_train_end].to_numpy()
    x_test = df_full[x_test_start:x_test_end].to_numpy()

    if should_scale:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)

    return x_train, x_test, config
