"""
Functions used to clean data

Gian Favero and Michael Montemurri, Mila, 2024    

Built on top of the work done by the NFL Big Data Bowl 2024 submission "Uncovering Tackle Opportunities and Missed Opportunities" that is 
created and maintained by @mpchang, @katdai, @bolongcheng, @danielrjiang
"""

import polars as pl


def aggregate_data(
    players_fname: str,
    plays_fname: str, 
    player_play_fname: str,
    games_fname: str,
    tracking_fname_list: list
) -> pl.DataFrame:
    """
    Create the aggregate dataframe by merging together the plays data and tracking data.

    :param plays_fname: the filename of the plays data
    :param tracking_fname_list: a list of filenames of the tracking data

    :return df_agg: the aggregate dataframe
    """
    print(
        "INFO: Aggregating data from players, play data, tracking data, and players data into a master dataframe..."
    )

    df_players = pl.read_csv(
        players_fname,
        null_values="NA",
    )
    df_plays = pl.read_csv(
        plays_fname,
        null_values="NA",
    )
    df_player_plays = pl.read_csv(
        player_play_fname,
        null_values="NA",
    )
    df_games = pl.read_csv(
        games_fname,
        null_values="NA",
    )
    df_tracking = pl.concat(
        [pl.read_csv(tracking_fname, null_values="NA") for tracking_fname in tracking_fname_list]
    )
    print(f"INFO: Loaded {len(df_plays)} rows of plays, {len(df_player_plays)} rows of player plays, and {len(df_tracking)} rows of player tracking data")

    # Aggregate player plays and players
    assert "nflId" in df_player_plays.columns, "ERROR: nflId column not found in player_plays dataframe"
    assert "nflId" in df_players.columns, "ERROR: nflId column not found in players dataframe"
    df_agg = df_player_plays.join(df_players, on=["nflId"], how="left")

    # Aggregate (player plays + players) and tracking data
    assert "gameId" in df_agg.columns, "ERROR: gameId column not found in player_plays dataframe"
    assert "playId" in df_agg.columns, "ERROR: playId column not found in player_plays dataframe"
    assert "gameId" in df_tracking.columns, "ERROR: gameId column not found in tracking dataframe"
    assert "playId" in df_tracking.columns, "ERROR: playId column not found in tracking dataframe"
    df_agg = df_agg.join(df_tracking, on=["gameId", "playId", "nflId"], how="left")

    # Aggregate (player plays + players + tracking) and plays
    assert "gameId" in df_agg.columns, "ERROR: gameId column not found in player_plays dataframe"
    assert "playId" in df_agg.columns, "ERROR: playId column not found in player_plays dataframe"
    assert "gameId" in df_plays.columns, "ERROR: gameId column not found in plays dataframe"
    assert "playId" in df_plays.columns, "ERROR: playId column not found in plays dataframe"
    df_agg = df_agg.join(df_plays, on=["gameId", "playId"], how="left")
    print(f"INFO: Aggregated dataframe has {len(df_agg)} rows")

    # Aggregate (player plays + players + tracking + plays) and games
    assert "gameId" in df_agg.columns, "ERROR: gameId column not found in player_plays dataframe"
    assert "gameId" in df_games.columns, "ERROR: gameId column not found in games dataframe"
    df_agg = df_agg.join(df_games, on=["gameId"], how="left")
    
    return df_agg


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


def remove_qb_kneels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows from the dataframe where playDescription contains "kneels" because these are not helpful for our model

    :param df (pl.DataFrame): dataframe to filter

    :return (pl.DataFrame): filtered dataframe with plays containing kneels are dropped
    """
    print("INFO: Removing QB kneels, spikes, sneaks...")
    
    # Print the unique values in qbKneel
    before = len(df)
    df = df.filter(pl.col("qbKneel") == 0) 
    df = df.filter(~pl.col("qbSpike") | (pl.col("qbSpike")).is_null())
    df = df.filter(~pl.col("qbSneak") | (pl.col("qbSneak")).is_null())
    after = len(df)
    print(f"INFO: {before - after} rows removed")

    return df


def remove_inactive_frames(df: pl.DataFrame, active_frames: str) -> pl.DataFrame:
    """
    Remove frames before ball_snap and after line_set for each play. 
    Group by gameId and playId to keep frames between ball_snap and line_set.

    Args:
        df (pl.DataFrame): DataFrame of tracking plays

    Returns:
        pl.DataFrame: DataFrame with frames between ball_snap and line_set
    """

    print("INFO: Removing inactive frames...")
    before = len(df)

    if active_frames == "at_snap":
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
    elif active_frames == "presnap":
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
    elif active_frames == "postsnap":
        # Step 1: Identify the indices of 'line_set' and 'tackle' for each gameID and playID
        # Filter to get the rows containing 'line_set' and 'tackle'
        events = df.filter(pl.col("event").is_in(["ball_snap", "tackle"]))

        # Step 2: For each group (gameID, playID), get the frameID corresponding to the 'line_set' and 'tackle'
        line_set_tackle_frames = events.group_by(["gameId", "playId"]).agg([
            pl.col("frameId").filter(pl.col("event") == "ball_snap").min().alias("ball_snap_frameID"),
            pl.col("frameId").filter(pl.col("event") == "tackle").max().alias("tackle_frameID")
        ])

        # Step 3: Join the 'ball_snap_frameID' back to the original dataframe on gameID and playID
        df_joined = df.join(line_set_tackle_frames, on=["gameId", "playId"], how="left")

        # Step 4: Filter the rows to only keep those where frameID is between ball_snap and tackle (inclusive)
        filtered = df_joined.filter(
            (pl.col("frameId") >= pl.col("ball_snap_frameID")) & 
            (pl.col("frameId") <= pl.col("tackle_frameID"))
        )
    elif active_frames == "all":
        # Step 1: Identify the indices of 'line_set' and 'tackle' for each gameID and playID
        # Filter to get the rows containing 'line_set' and 'tackle'
        events = df.filter(pl.col("event").is_in(["line_set", "tackle"]))

        # Step 2: For each group (gameID, playID), get the frameID corresponding to the 'line_set' and 'tackle'
        line_set_tackle_frames = events.group_by(["gameId", "playId"]).agg([
            pl.col("frameId").filter(pl.col("event") == "line_set").min().alias("line_set_frameID"),
            pl.col("frameId").filter(pl.col("event") == "tackle").max().alias("tackle_frameID")
        ])

        # Step 3: Join the 'line_set_tackle_frames' back to the original dataframe on gameID and playID
        df_joined = df.join(line_set_tackle_frames, on=["gameId", "playId"], how="left")

        # Step 4: Filter the rows to only keep those where frameID is between line_set and tackle (inclusive)
        filtered = df_joined.filter(
            (pl.col("frameId") >= pl.col("line_set_frameID")) & 
            (pl.col("frameId") <= pl.col("tackle_frameID"))
        )
    else:
        raise ValueError("Invalid argument for active_frames. Must be one of ['presnap', 'postsnap', 'all']")
    after = len(filtered)
    print(f"INFO: {before - after} rows removed")

    return filtered


def remove_garbage_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove frames in which either the home or visitor team has a >95% win probability

    Args:
        df (pl.DataFrame): DataFrame of tracking plays

    Returns:
        pl.DataFrame: DataFrame with frames in garbage time removed
    """

    print("INFO: Removing garbage time frames...")

    before = len(df)
    df = df.filter(
        ~(pl.col("preSnapHomeTeamWinProbability") > 0.95)
    )
    df = df.filter(
        ~(pl.col("preSnapVisitorTeamWinProbability") > 0.95)
    )
    after = len(df)
    print(f"INFO: {before - after} rows removed")

    return df

def strip_unused_data(df: pl.DataFrame, useful_columns: list) -> pl.DataFrame:
    """
    Drop dataframe columns from df that aren't useful in actually training our models, to make memory usage smaller.

    :param df (pl.DataFrame): dataframe to filter
    """
    # Only keep columns critical for the PlayFrame class to function
    print("INFO: Removing unused columns from dataframe...")
    before = len(df.columns)
    df = df.select([col for col in useful_columns if col in df.columns])
    after = len(df.columns)
    print(f"INFO: {before - after} columns removed")

    return df


def clean_data(df: pl.DataFrame, active_frames: str) -> pl.DataFrame:
    """
    Takes as input the aggregated dataframe of plays, tackles, players, and tracking data and performs
    the following preprocessing operations:

    1) Rotates the direction and orientation data so that the convention matches the unit circle
    2) Flips plays so that they run from left to right

    Subsequently, it cleans the data as follows:

    1) Remove inactive frames (before the ball snap and after the tackle)
    2) Strip unused df columns to save memory

    :param df (pl.DataFrame): the original, aggregated dataframe
    :return df_clean (pl.DataFrame): the cleaned dataframe
    """
    # Data Cleaning
    df = remove_inactive_frames(df, active_frames)
    df = remove_garbage_time(df)

    # Data preprocessing so that all plays run from left-to-right and all angles match the standard unit circle convention
    df = rotate_direction_and_orientation(df)
    df = make_plays_left_to_right(df)

    # Get rid of penalties
    df = remove_qb_kneels(df)
    # Optimize memory usage
    df = convert_geometry_to_int(df)

    return df