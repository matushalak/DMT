import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocess import preprocess_pipeline

def engineer_features(preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: add sleep
    # TODO: difference - on day AND time of day level
    # TODO: diff_agg = ["mood", "circumplex.valence", "circumplex.arousal"]
    breakpoint()


if __name__ == '__main__':
    data = preprocess_pipeline(load_from_file=True)
    engineer_features(data)
