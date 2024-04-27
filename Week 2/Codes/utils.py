import re
import sys
import copy
import logging

import numpy as np
import pandas as pd

from pyarrow import parquet as pq

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from aim.catboost import AimLogger
#from aim.experiment_tracker.catboost import Logger as AimLogger
from sys import stdout
from typing import Optional, Dict, List
from aim import Run
from aim.ext.resource import DEFAULT_SYSTEM_TRACKING_INT


class AimLogger:
    """
    AimLogger logger class.

    Args:
        repo (:obj:`str`, optional): Aim repository path or Repo object to which Run object is bound.
            If skipped, default Repo is used.
        experiment_name (:obj:`str`, optional): Sets Run's `experiment` property. 'default' if not specified.
            Can be used later to query runs/sequences.
        system_tracking_interval (:obj:`int`, optional): Sets the tracking interval in seconds for system usage
            metrics (CPU, Memory, etc.). Set to `None` to disable system metrics tracking.
        log_system_params (:obj:`bool`, optional): Enable/Disable logging of system params such as installed packages,
            git info, environment variables, etc.
        capture_terminal_logs (:obj:`bool`, optional): Enable/Disable terminal stdout logging.
        loss_function (:obj:`str`, optional): Loss function
        log_cout (:obj:`bool`, optional): Enable/Disable stdout logging.
    """

    def __init__(
        self,
        hyper_params: Optional[dict],
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        loss_function: Optional[str] = 'Loss',
        log_cout=stdout,
    ):
        super().__init__()
        self._repo_path = repo
        self._experiment = experiment
        self._system_tracking_interval = system_tracking_interval
        self._log_system_params = log_system_params
        self._capture_terminal_logs = capture_terminal_logs
        self._run = None
        self._run_hash = None
        self._loss_function = loss_function
        self._log_cout = log_cout
        self._hyper_params = hyper_params
        
        if log_cout is not None:
            assert hasattr(log_cout, 'write')

    @property
    def experiment(self) -> Run:
        if not self._run:
            self.setup()
        return self._run

    def setup(self):
        if self._run:
            return
        if self._run_hash:
            self._run = Run(
                self._run_hash,
                repo=self._repo_path,
                system_tracking_interval=self._system_tracking_interval,
                capture_terminal_logs=self._capture_terminal_logs,
            )
        else:
            self._run = Run(
                repo=self._repo_path,
                experiment=self._experiment,
                system_tracking_interval=self._system_tracking_interval,
                log_system_params=self._log_system_params,
                capture_terminal_logs=self._capture_terminal_logs,
            )
            self._run_hash = self._run.hash
            self._run['hparams']= self._hyper_params

    def _to_number(self, val):
        try:
            return float(val)
        except ValueError:
            return val

    def write(self, log):
        run = self.experiment

        _log = log
        log = log.strip().split()
        if log:
            if len(log) == 3 and log[1] == '=':
                run[log[0]] = self._to_number(log[2])
                return

            value_learn = None
            value_iter = None
            value_test = None
            value_best = None

            if log[1] == 'learn:':
                value_iter = int(log[0][:-1])
                value_learn = self._to_number(log[2])
                if log[3] == 'test:':
                    value_test = self._to_number(log[4])
                    if log[5] == 'best:':
                        value_best = self._to_number(log[6])
            if any((value_learn, value_test, value_best)):
                if value_learn:
                    run.track(
                        value_learn,
                        name=self._loss_function,
                        step=value_iter,
                        context={'log': 'learn'},
                    )
                if value_test:
                    run.track(
                        value_test,
                        name=self._loss_function,
                        step=value_iter,
                        context={'log': 'test'},
                    )
                if value_best:
                    run.track(
                        value_best,
                        name=self._loss_function,
                        step=value_iter,
                        context={'log': 'best'},
                    )
            else:
                # Unhandled or junky log
                pass

        if self._log_cout:
            self._log_cout.write(_log)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# Reader Function
def data_reader(cols, path):
    
    df= pq.read_table(source = path,
                      columns= cols).to_pandas()
    
    return df

# Check Nulls
def check_nulls(dataframe):
    
    # Turning ? to nulls
    dataframe = dataframe.replace({'?': np.nan})

    # Studying the nulls percentage for each column
    nulls_info = (dataframe.isnull().sum()/dataframe.shape[0] * 100).sort_values(ascending=False)
    logger.info(f'Nulls_info:\n{nulls_info}')

def age_dict(age_range):
    
    age_values = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, 
                  '[30-40)': 35, '[40-50)': 45, '[50-60)': 55, 
                  '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, 
                  '[90-100)': 95}
    
    if age_range in age_values.keys():
        value = age_values[age_range]
    else: 
        value = np.nan
    return value
    
def max_glu_serum_dict(max_glu_serum_range):
    
    max_glu_serum_values = {'None': 0, 
                            'Norm': 1, 
                            '>200': 2, 
                            '>300': 3}
    
    if max_glu_serum_range in max_glu_serum_values.keys():
        value = max_glu_serum_values[max_glu_serum_range]
    else: value = np.nan
    
    return value

def a1cresult_dict(a1cresult_range):
    
    a1cresult_values = {'None': 0, 
                        'Norm': 1, 
                        '>7': 2, 
                        '>8': 3}
    
    if a1cresult_range in a1cresult_values.keys():
        value = a1cresult_values[a1cresult_range]
    else: value = np.nan
    
    return value

 
def load_model(model_path):
    
    model= CatBoostClassifier()
    model.load_model(model_path)
    
    return model

def impute_race_column(df):
    
    known_patients = df[~pd.isnull(df.race)][['patient_nbr', 'race']].drop_duplicates()
    
    logger.info('The number of nulls on the race column before this step is {}'.format(sum(pd.isnull(df.race))))
    
    known_patients_n_races = known_patients.groupby('patient_nbr').count()
    
    invalid_patient_nbr = list(known_patients_n_races[known_patients_n_races.race != 1].index)
    known_patients = known_patients[~known_patients.patient_nbr.isin(invalid_patient_nbr)]
    
    # We can already join it with the df df. We already have its primary key ready.
    df = pd.merge(df, known_patients, how='left', on='patient_nbr')
    
    # If we don't know the race using race_x (original), we use race_y (calculated)
    df['race'] = df['race_x'].combine_first(df['race_y'])
    df = df.drop(columns=['race_x', 'race_y'])
    
    logger.info('The number of nulls on the race column after this step is {}'.format(sum(pd.isnull(df.race))))

    imp_model= load_model('artifacts/imputer_model.cbm')
    
    df.loc[df.race.isna(),'race']= imp_model.predict(df.loc[df.race.isna(),imp_model.feature_names_])
    
    logger.info('The number of nulls on the race column after last step is {}'.format(sum(pd.isnull(df.race))))
        
    return df

def train(df):

    #Define Target and Inputs
    target = 'readmitted'
    not_input = ['readmitted', 'patient_nbr', 'encounter_id']
    input_features = [column for column in df.columns if column not in not_input]
    cat_features= list(df[input_features].select_dtypes(include='object').columns)

    #Define Train Test Set
    y = df.readmitted
    x = df[input_features]

    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        train_size= 0.8, 
                                                        stratify = y,
                                                        random_state=10)
    X_train.to_parquet('Train_set.pq')
    # Define pools
    
    pool_train= Pool(data= X_train,
                     label=y_train,
                     cat_features=cat_features)
    
    pool_test=  Pool(data= X_test,
                     label=y_test,
                     cat_features=cat_features)
    params= {'max_depth':5,
            'learning_rate':0.01,
            'random_state':1,
            'eval_metric':'Accuracy'}
    
    model= CatBoostClassifier(**params)
    model.fit(pool_train,
              use_best_model=True,
#              verbose=100,
              eval_set=pool_test,
              
#              log_cout=AimLogger(loss_function='Accuracy'), logging_level='Verbose')
              log_cout=AimLogger(loss_function='Accuracy'
                                 ,hyper_params= params
                                )) 
#              logging_level='Verbose             )

    return model
