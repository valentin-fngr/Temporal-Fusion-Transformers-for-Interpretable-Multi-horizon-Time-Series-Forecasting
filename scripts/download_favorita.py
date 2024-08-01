import os 
import gc
import glob 

import pandas as pd 
import numpy as np 
import pyunpack



default_config = {
  "data_folder": "./datasets/favorita/", # data folder with zip files. 
  "csv_path" : "./datasets/favorita/"

}

def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))


def process_favorita(config):
  """Processes Favorita dataset.

  Makes use of the raw files should be manually downloaded from Kaggle @
    http://www.kaggle.com/c/favorita-grocery-sales-forecasting/data

  Args:
    config: Default experiment config for Favorita
  """


  data_folder = config["data_folder"]

  zip_file = os.path.join(data_folder, 'favorita-grocery-sales-forecasting.zip')

  # Unpack main zip file
  outputs_file = os.path.join(data_folder, 'train.csv.7z')
  unzip(zip_file, outputs_file, data_folder)

  # Unpack individually zipped files
  # for file in glob.glob(os.path.join(data_folder, '*.7z')):

  #   csv_file = file.replace('.7z', '')

  #   unzip(file, csv_file, data_folder)

  print('Unzipping complete, commencing data processing...')

  # Extract only a subset of data to save/process for efficiency
  start_date = pd.to_datetime("2015-01-01")
  end_date = pd.to_datetime("2016-6-1")

  print('Regenerating data...')

  # load temporal data
  print("read temporal ...")
  temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0, low_memory=True)

  print("read store info ...")
  store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0, low_memory=True)
  print("Read oil ...")
  oil = pd.read_csv(
      os.path.join(data_folder, 'oil.csv'), index_col=0, low_memory=True).iloc[:, 0], 
      
  print("read holidays ...")
  holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'), low_memory=True)
  print("read itens ....")
  items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0, low_memory=True)
  print("read transactions ...")
  transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'), low_memory=True)

  # Take first 6 months of data
  temporal['date'] = pd.to_datetime(temporal['date'])

  # Filter dates to reduce storage space requirements
  if start_date is not None:
    temporal = temporal[(temporal['date'] >= start_date)]
  if end_date is not None:
    temporal = temporal[(temporal['date'] < end_date)]

  dates = temporal['date'].unique()

  # Add trajectory identifier
  temporal['traj_id'] = temporal['store_nbr'].apply(
      str) + '_' + temporal['item_nbr'].apply(str)
  temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(
      str)

  # Remove all IDs with negative returns
  print('Removing returns data')
  min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
  valid_ids = set(min_returns[min_returns >= 0].index)
  selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
  new_temporal = temporal[selector].copy()
  del temporal
  gc.collect()
  temporal = new_temporal
  temporal['open'] = 1

  # Resampling
  print('Resampling to regular grid')
  resampled_dfs = []
  for traj_id, raw_sub_df in temporal.groupby('traj_id'):
    print('Resampling', traj_id)
    sub_df = raw_sub_df.set_index('date', drop=True).copy()
    sub_df = sub_df.resample('1d').last()
    sub_df['date'] = sub_df.index
    sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
        = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
    sub_df['open'] = sub_df['open'].fillna(
        0)  # flag where sales data is unknown
    sub_df['log_sales'] = np.log(sub_df['unit_sales'])

    resampled_dfs.append(sub_df.reset_index(drop=True))

  new_temporal = pd.concat(resampled_dfs, axis=0)
  del temporal
  gc.collect()
  temporal = new_temporal

  print('Adding oil')
  oil.name = 'oil'
  oil.index = pd.to_datetime(oil.index)
  temporal = temporal.join(
      oil.loc[dates].fillna(method='ffill'), on='date', how='left')
  temporal['oil'] = temporal['oil'].fillna(-1)

  print('Adding store info')
  temporal = temporal.join(store_info, on='store_nbr', how='left')

  print('Adding item info')
  temporal = temporal.join(items, on='item_nbr', how='left')

  transactions['date'] = pd.to_datetime(transactions['date'])
  temporal = temporal.merge(
      transactions,
      left_on=['date', 'store_nbr'],
      right_on=['date', 'store_nbr'],
      how='left')
  temporal['transactions'] = temporal['transactions'].fillna(-1)

  # Additional date info
  temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
  temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
  temporal['month'] = pd.to_datetime(temporal['date'].values).month

  # Add holiday info
  print('Adding holidays')
  holiday_subset = holidays[holidays['transferred'].apply(
      lambda x: not x)].copy()
  holiday_subset.columns = [
      s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
  ]
  holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
  local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
  regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
  national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

  temporal['national_hol'] = temporal.merge(
      national_holidays, left_on=['date'], right_on=['date'],
      how='left')['description'].fillna('')
  temporal['regional_hol'] = temporal.merge(
      regional_holidays,
      left_on=['state', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')
  temporal['local_hol'] = temporal.merge(
      local_holidays,
      left_on=['city', 'date'],
      right_on=['locale_name', 'date'],
      how='left')['description'].fillna('')

  temporal.sort_values('unique_id', inplace=True)

  print('Saving processed file to {}'.format(config.data_csv_path))
  temporal.to_csv(config.data_csv_path)



if __name__ == "__main__": 
  process_favorita(default_config)