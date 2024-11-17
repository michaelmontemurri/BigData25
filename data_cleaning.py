"""
Functions used to clean data    
"""

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from constants import CLUB_DICT

import polars as pl


def aggregate_data(
    plays_fname: str, 
    player_plays_fname: str,
    tracking_fname_list: list
) -> pl.DataFrame:
    """
    Create the aggregate dataframe by merging together the plays data and tracking data.

    :param plays_fname: the filename of the plays data
    :param tracking_fname_list: a list of filenames of the tracking data

    :return df_agg: the aggregate dataframe
    """
    print(
        "INFO: Aggregating data from play data, tracking data, and players data into a master dataframe..."
    )

    df_plays = pl.read_csv(
        plays_fname,
        null_values="NA",
    )
    df_player_plays = pl.read_csv(
        player_plays_fname,
        null_values="NA",
    )
    df_tracking = pl.read_csv(
        tracking_fname_list[0],
        null_values="NA",
    )
    #df_tracking = pl.concat(
    #    [pl.read_csv(tracking_fname, null_values="NA") for tracking_fname in tracking_fname_list]
    #)

    # Aggregate plays and tracking
    df_agg = df_player_plays.join(df_tracking, on="nflId", how="inner")
    df_agg1 = df_agg.join(df_tracking, on=["gameId", "playId"], how="inner")

    return df_agg1


def rotate_direction_and_orientation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rotate the direction and orientation angles so that 0° points from left to right on the field, and increasing angle goes counterclockwise
    This should be done BEFORE the call to make_plays_left_to_right, because that function with compensate for the flipped angles.

    :param df: the aggregate dataframe created using the aggregate_data() method

    :return df: the aggregate dataframe with orientation and direction angles rotated 90° clockwise
    """
    print(
        "INFO: Transforming orientation and direction angles so that 0° points from left to right, and increasing angle goes counterclockwise..."
    )

    df = df.with_columns([
        (-(pl.col("o") - 90) % 360).alias("o_clean"),
        (-(pl.col("dir") - 90) % 360).alias("dir_clean")
    ])
    
    return df


def make_plays_left_to_right(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flip tracking data so that all plays run from left to right. The new x, y, s, a, dis, o, and dir data
    will be stored in new columns with the suffix "_clean" even if the variables do not change from their original value.

    :param df: the aggregate dataframe

    :return df: the aggregate dataframe with the new columns such that all plays run left to right
    """
    print("INFO: Flipping plays so that they all run from left to right...")

    df = df.with_columns([
        # x_clean: if playDirection is "left", calculate 120 - x; otherwise, keep x
        pl.when(pl.col("playDirection") == "left")
          .then(120 - pl.col("x"))
          .otherwise(pl.col("x"))
          .alias("x_clean"),

        # y_clean, s_clean, a_clean, and dis_clean are just copied over without modification
        pl.col("y").alias("y_clean"),
        pl.col("s").alias("s_clean"),
        pl.col("a").alias("a_clean"),
        pl.col("dis").alias("dis_clean"),

        # o_clean: if playDirection is "left", calculate 180 - o; otherwise, keep o
        ((pl.when(pl.col("playDirection") == "left")
            .then(180 - pl.col("o"))
            .otherwise(pl.col("o"))
          + 360) % 360).alias("o_clean"),

        # dir_clean: if playDirection is "left", calculate 180 - dir; otherwise, keep dir
        ((pl.when(pl.col("playDirection") == "left")
            .then(180 - pl.col("dir"))
            .otherwise(pl.col("dir"))
          + 360) % 360).alias("dir_clean")
    ])

    return df


def convert_geometry_to_int(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert the x_clean, y_clean, dir_clean, o_clean, s_clean, a_clean columns to int to reduce dataframe size.
    We do this by multiplying the position, speed, acceleration vectors by 100, and the angle vectors by 10, and
    rounding to the nearest integer. This effectively reduces the precision of position, speed, and acceleration
    to the hundredths decimal, and the angle to the tenth decimal.

    :param df: the aggregate dataframe created using the aggregate_data() method

    :return df: the aggregate dataframe with the geometry column converted to a tuple of ints
    """
    state_cols = ["x_clean", "y_clean", "s_clean", "a_clean"]
    angle_cols = ["dir_clean", "o_clean"]

    print("INFO: Converting geometry variables from floats to int...")
    
    # Round and scale the state columns (position, speed, acceleration)
    for col in state_cols:
        df = df.with_columns(
            (pl.col(col) * 100).round().alias(col)
        )
        # Assert max value is within the acceptable range for int16
        max_value = df.select(pl.col(col).max()).to_pandas().iloc[0, 0]
        assert abs(max_value) < 32767, f"ERROR: The max value of column {col} is too large for int16"

    # Round and scale the angle columns
    for col in angle_cols:
        df = df.with_columns(
            (pl.col(col) * 10).round().alias(col)
        )
        # Assert values are non-negative and within the acceptable range
        min_value = df[col].min()
        assert min_value >= 0, "Angles should be greater than 0"
        max_value = df[col].max()
        if max_value > 32767:
            print(f"WARNING: The max value of column {col} is too large for int16")
        assert max_value < 32767, f"ERROR: The max value of column {col} is too large for int16"
    
    # Cast columns to int16
    df = df.with_columns(
        [pl.col(col).cast(pl.Int16).alias(col) for col in state_cols + angle_cols]
    )

    return df

def remove_plays_with_penalties(df, strict=False):
    """
    Remove rows from the dataframe where playNullifiedByPenalty == "Y" because these are not helpful for our model

    :param: df (pl.DataFrame): dataframe to filter
    :param: strict (boolean): if False, only drop plays where playNullifiedByPenalty == "Y". If True, drop plays where foulName1 is not NaN

    :return: (pl.DataFrame) filtered dataframe with plays nullified by penalties are dropped
    """
    print("INFO: Removing play with penalties...")
    before = len(df)
    if strict:
        df = df[(df.foulName1.isna()) & (df.playNullifiedByPenalty != "Y")]
    else:
        df = df[df.playNullifiedByPenalty != "Y"]
    after = len(df)
    print(f"INFO: {before - after} rows removed")
    return df


def remove_touchdowns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove touchdowns from the dataframe, since these are not relevant for tackling

    :param: df (pl.DataFrame): dataframe to filter

    :return (pl.DataFrame) filtered dataframe with all frames associated with touchdown plays removed
    """
    print("INFO: Removing plays with touchdowns")
    before = len(df)
    td_plays = []

    df_td = df[df.event == "touchdown"]
    td_plays = df_td.groupby(["gameId", "playId"]).groups.keys()

    for gameId, playId in tqdm(td_plays, ascii=True, desc="remove_touchdowns"):
        df = df[~((df.gameId == gameId) & (df.playId == playId))]

    after = len(df)
    print(f"INFO: {before - after} rows removed")

    return df


def remove_inactive_frames1(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove frames before ball_snap and after line_set for each play. 
    Group by gameId and playId to keep frames between ball_snap and line_set.

    Args:
        df (pl.DataFrame): DataFrame of tracking plays

    Returns:
        pl.DataFrame: DataFrame with frames between ball_snap and line_set
    """

    print("INFO: Removing inactive frames...")

    # Step 1: Identify the indices of 'line_set' and 'ball_snapped' for each gameID and playID
    # Filter to get the rows containing 'line_set' and 'ball_snapped'
    events = df.filter(pl.col("event").is_in(["line_set", "ball_snap"]))

    # Step 2: For each group (gameID, playID), get the frameID corresponding to the 'line_set' and 'ball_snapped'
    line_set_ball_snapped_frames = events.group_by(["gameId", "playId"]).agg([
        pl.col("frameId").filter(pl.col("event") == "line_set").min().alias("line_set_frameID"),
        pl.col("frameId").filter(pl.col("event") == "ball_snap").max().alias("ball_snapped_frameID")
    ])

    # Step 3: Join the 'line_set_ball_snapped_frames' back to the original dataframe on gameID and playID
    df_joined = df.join(line_set_ball_snapped_frames, on=["gameId", "playId"], how="left")

    # Step 4: Filter the rows to only keep those where frameID is between line_set and ball_snapped (inclusive)
    filtered = df_joined.filter(
        (pl.col("frameId") >= pl.col("line_set_frameID")) & 
        (pl.col("frameId") <= pl.col("ball_snapped_frameID"))
    )

    return filtered

def remove_inactive_frames(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove frames before ball_snap and after line_set for each play. 
    Group by gameId and playId to keep frames between ball_snap and line_set.

    Args:
        df (pl.DataFrame): DataFrame of tracking plays

    Returns:
        pl.DataFrame: DataFrame with frames between ball_snap and line_set
    """

    print("INFO: Removing inactive frames...")

    # Step 1: Identify the indices of 'line_set' and 'ball_snapped' for each gameID and playID
    # Filter to get the rows containing 'line_set' and 'ball_snapped'
    events = df.filter(pl.col("event").is_in(["line_set", "ball_snap"]))

    # Step 2: For each group (gameID, playID), get the frameID corresponding to the 'line_set' and 'ball_snapped'
    line_set_ball_snapped_frames = events.group_by(["gameId", "playId"]).agg([
        pl.col("frameId").filter(pl.col("event") == "line_set").min().alias("line_set_frameID"),
        pl.col("frameId").filter(pl.col("event") == "ball_snap").max().alias("ball_snapped_frameID")
    ])

    # Step 3: Join the 'line_set_ball_snapped_frames' back to the original dataframe on gameID and playID
    df_joined = df.join(line_set_ball_snapped_frames, on=["gameId", "playId"], how="left")

    # Step 4: Filter the rows to only keep those where frameID is between line_set and ball_snapped (inclusive)
    filtered = df_joined.filter(
        (pl.col("frameId") == pl.col("line_set_frameID")) |
        (pl.col("frameId") == pl.col("ball_snapped_frameID"))
    )

    return filtered


def strip_unused_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop dataframe columns from df that aren't useful in actually training our models, to make memory usage smaller.

    :param df (pl.DataFrame): dataframe to filter
    """

    # Print the columns that are in the dataframe
    print("INFO: Columns in the dataframe:")
    print(df.columns)

    '''# Only keep columns critical for the PlayFrame class to function
    useful_columns = [
        "gameId",
        "playId",
        "frameId",
        "club",
        "possessionTeam",
        "defensiveTeam",
        "is_run",
        "nflId",
        "o_clean",
        "a_clean",
        "s_clean",
        "x_clean",
        "y_clean",
        "dir_clean",
        "weight",
        "ballCarrierId",
        "playResult",
        "event",
        "tackle_dict",
        "age",
    ]
    columns_to_drop = [col for col in df.columns if col not in useful_columns]
    print("INFO: Removing unused columns from dataframe...")
    before = len(df.columns)
    df = df.drop(columns=columns_to_drop)
    after = len(df.columns)
    print(f"INFO: {before - after} columns removed")'''
    return df


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Takes as input the aggregated dataframe of plays, tackles, players, and tracking data and performs
    the following preprocessing operations:

    1) Rotates the direction and orientation data so that the convention matches the unit circle
    2) Flips plays so that they run from left to right
    3) Adds a label to indicate whether the play is a pass or a run

    Subsequently, it cleans the data as follows:

    1) Remove plays with penalties
    2) Remove plays that resulted in touchdowns
    3) Convert teams from strings to ints to reduce memory
    4) Remove inactive frames (before the ball snap and after the tackle)
    5) Remove any bad data (not all players are tracked, multiple ballcarriers)
    6) Strip unused df columns to save memory

    :param df (pl.DataFrame): the original, aggregated dataframe
    :return df_clean (pl.DataFrame): the cleaned dataframe
    """

    # Data preprocessing so that all plays run from left-to-right and all angles match the standard unit circle convention
    df = rotate_direction_and_orientation(df)
    df = make_plays_left_to_right(df)

    # Data Cleaning
    #df = remove_plays_with_penalties(df, strict=True)
    #df = remove_touchdowns(df)
    df = strip_unused_data(df)
    df = remove_inactive_frames(df)

    # Only keep game ID for now
    filtered_df = df.filter(pl.col("gameId") == 2022091200)

    # Write to csv
    filtered_df.write_csv("data/cleaned_data.csv")

    return df


def optimize_memory_usage(df):
    """
    Optimize the memory usage by performing the following numerical operations:

    1) Converts the speed and position to ints by multiplying by 100 and then converting to int
    2) Converts angles to ints by multiplying by 10 and then converting to int
    """

    df = convert_geometry_to_int(df)

    return df