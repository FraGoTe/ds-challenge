import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Union, List
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from pathlib import Path
from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        #load a saved model by default
        self._model_path = 'data/xgboost_model.pkl'
        self.load_model(self._model_path)

        self.top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def get_top_10_features(self) -> list:
        return self.top_10_features

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['period_day'] = data['Fecha-I'].apply(self._get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self._is_high_season)
        data['min_diff'] = data.apply(self._get_min_diff, axis=1)

        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
       
        features = self.feature_format(data)
        target = data['delay']
        
        return features, target

    def feature_format(data: pd.DataFrame) -> pd.DataFrame:
        return  pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Calculate the scale to handle imbalance in the target variable
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1
        # Define and train the XGBoost model
        self._model = XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)

        # Save the trained model in a file
        self.save_model(filepath = self._model_path)

    def split_data(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> List:
        x_train2, x_test2, y_train2, y_test2 = train_test_split(features[self.top_10_features], target, test_size = 0.30, random_state = 42)

        return x_train2, x_test2, y_train2, y_test2


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if not self._model:
            raise ValueError("Model has not been loaded or trained yet.")
        return self._model.predict(features).tolist()

    def save_model(self, filepath: str) -> None:
        """
        Save the model's parameters and attributes into a file.

        Args:
            filepath (str): Path to the file where the model should be saved.
        """
        joblib.dump(self._model, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load the model from the file and set it to the self._model attribute.

        Args:
            filepath (str): Path to the file to load the model from.
        """
        if Path(filepath).exists():
            self._model = joblib.load(filepath)
        else:
            print(f"Model file {filepath} does not exist. Please train the model first.")

    @staticmethod
    def _get_period_day(date: str) -> str:
        """
        Determine the period of the day for a flight's timestamp.

        Args:
            date (str): The timestamp string in the format '%Y-%m-%d %H:%M:%S'.

        Returns:
            str: The period of the day (mañana, tarde, noche).
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("04:59", '%H:%M').time()

        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        elif evening_min <= date_time <= evening_max or night_min <= date_time <= night_max:
            return 'noche'
        else:
            return 'noche'

    @staticmethod
    def _is_high_season(fecha: str) -> int:
        """
        Determine if a flight's date falls within the high season periods.

        Args:
            fecha (str): The timestamp string in the format '%Y-%m-%d %H:%M:%S'.

        Returns:
            int: 1 if high season, 0 otherwise.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    @staticmethod
    def _get_min_diff(data: pd.Series) -> float:
        """
        Calculate the difference in minutes between the scheduled and actual flight times.

        Args:
            data (pd.Series): A row from the DataFrame containing 'Fecha-O' and 'Fecha-I'.

        Returns:
            float: Difference in minutes between 'Fecha-O' and 'Fecha-I'.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return (fecha_o - fecha_i).total_seconds() / 60