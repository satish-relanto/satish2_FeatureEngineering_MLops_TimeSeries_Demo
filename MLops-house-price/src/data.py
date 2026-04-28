import pandas as pd
from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing


def load_data(as_frame=True):
    try:
        data=fetch_california_housing(as_frame=True)
        X=data.frame.drop(columns=["MedHouseVal"]) if as_frame else data.data
        y=data.frame["MedHouseVal"] if as_frame else data.target
    except Exception as exc:
        print(f"Could not load California housing dataset, using fallback data: {exc}")
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        X_array,y_array=make_regression(
            n_samples=1000,
            n_features=len(columns),
            noise=0.2,
            random_state=42,
        )
        if as_frame:
            X=pd.DataFrame(X_array,columns=columns)
            y=pd.Series(y_array,name="MedHouseVal")
        else:
            X=X_array
            y=y_array
    return X,y

if __name__=="__main__":
    X,y=load_data()
    print(f"Shape of X is {X.shape} and y is {y.shape}")

