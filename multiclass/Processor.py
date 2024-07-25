import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from lib.sampling import sampling
from mcfly import modelgen
from lib.find_architecture import train_models_on_samples 
from lib.data_preparation import TensorflowTsDataPreparer
from lib.merge_all_in_tolerance import merge_asof_outer
from services.BaseService import BaseService
from services.S3Service import S3Service
from services.SegmentService import SegmentService
from services.DataService import DataService

line_id = 'line_id'
actual_model = 'actual_model.h5'

log_accu_percentage = 'log_accu_percentage.number'
log_per_minute = 'log_per_minute.number'
log_produced2 = 'logproduced2.number'
log_rejected2 = 'logrejected2.number'
speed = 'speed.number'
numstops = 'numstops.number'
tension = 'tension.number'
tension_eb1 = 'tension_eb1.number'
tension_eb2 = 'tension_eb2.number'
tension_rw = 'tension_rw.number'
tension_uw1 = 'tension_uw1.number'
tension_uw2 = 'tension_uw2.number'
tension_uw3 = 'tension_uw3.number'


class Processor(BaseService):
    def __init__(self, s3_service, segment_service, data_service):
        BaseService.__init__(self)
        self.__s3_service = s3_service
        self.__segment_service = segment_service
        self.__data_service = data_service

    def get_10s_data(self):
        df_log_accu_percentage = self.__s3_service.to_df(line_id, log_accu_percentage)
        df_log_per_minute = self.__s3_service.to_df(line_id, log_per_minute)
        df_log_produced2 = self.__s3_service.to_df(line_id, log_produced2)
        df_log_rejected2 = self.__s3_service.to_df(line_id, log_rejected2)
        df_speed = self.__s3_service.to_df(line_id, speed)
        df_num_stops = self.__s3_service.to_df(line_id, numstops)
        df_tension = self.__s3_service.to_df(line_id, tension)
        df_tension_eb1 = self.__s3_service.to_df(line_id, tension_eb1)
        df_tension_eb2 = self.__s3_service.to_df(line_id, tension_eb2)
        df_tension_rw = self.__s3_service.to_df(line_id, tension_rw)
        df_tension_uw1 = self.__s3_service.to_df(line_id, tension_uw1)
        df_tension_uw2 = self.__s3_service.to_df(line_id, tension_uw2)
        df_tension_uw3 = self.__s3_service.to_df(line_id, tension_uw3)

        df_telemetry = df_log_accu_percentage.merge(df_log_per_minute, left_on=df_log_accu_percentage.columns[0], right_on=df_log_per_minute.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_log_produced2, left_on=df_telemetry.columns[0], right_on=df_log_produced2.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_log_rejected2, left_on=df_telemetry.columns[0], right_on=df_log_rejected2.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_speed, left_on=df_telemetry.columns[0], right_on=df_num_stops.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_num_stops, left_on=df_telemetry.columns[0], right_on=df_speed.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension, left_on=df_telemetry.columns[0], right_on=df_tension.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension_eb1, left_on=df_telemetry.columns[0], right_on=df_tension_eb1.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension_eb2, left_on=df_telemetry.columns[0], right_on=df_tension_eb2.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension_rw, left_on=df_telemetry.columns[0], right_on=df_tension_rw.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension_uw1, left_on=df_telemetry.columns[0], right_on=df_tension_uw1.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension_uw2, left_on=df_telemetry.columns[0], right_on=df_tension_uw2.columns[0], how='inner')
        df_telemetry = df_telemetry.merge(df_tension_uw3, left_on=df_telemetry.columns[0], right_on=df_tension_uw3.columns[0], how='inner')

        df_telemetry = df_telemetry.rename(columns={df_telemetry.columns[0]: "Date"})
        df_telemetry['Date'] = pd.to_datetime(df_telemetry['Date'], unit='ms')

        return df_telemetry


    def calculate_weights_and_upload(self):
        self.logger.info('Starting processing...')
        start_date = datetime.datetime(2018, 4, 12)
        end_date = datetime.datetime(2019, 3, 1)

        df_telemetry = self.get_10s_data()
        
        df_segments = self.__data_service.prepare_segments_with_cards(line_id, start_date, end_date)
        
        df_segments['time_diff'] = df_segments['endTime']-df_segments['startTime']
        df_segments['time_diff'] = df_segments['time_diff'].dt.total_seconds() / 60
        
        avg_segment = round(df_segments['time_diff'].mean())
        avg_segment = avg_segment.astype('int')
        percentile = np.percentile(df_segments['time_diff'], 99.5)

        telemetry_ts_name = 'Date'
        category_name = 'card.rootCause.id'
        id_name = 'id'
        segment_start_name = 'startTime'
        
        df_segments = df_segments[df_segments.time_diff<percentile]

        df_segments = df_segments[[id_name, segment_start_name, category_name]]

        labeled_segment_df = df_segments[~df_segments[category_name].isna()]
        
        time_before_downtime = pd.to_timedelta(avg_segment, unit='m')

        merged_df = merge_asof_outer(labeled_segment_df,
                                     df_telemetry,
                                     left_on=segment_start_name,
                                     right_on=telemetry_ts_name,
                                     tolerance=time_before_downtime,
                                     )

        merged_df = merged_df.rename(columns={category_name: 'category', telemetry_ts_name: 'ts',})
      
        max_category = merged_df['category'].value_counts().idxmax()
        max_category = max_category.split(',')
        merged_df.loc[~merged_df['category'].isin(max_category), 'category'] = 'other'
        merged_df = merged_df.dropna()
        
        ts_lengths_by_id = merged_df.groupby(id_name).size()
        num_measurements = avg_segment
        
        all_ids = ts_lengths_by_id[ts_lengths_by_id == num_measurements].index.values
        train_ids, test_ids = train_test_split(all_ids, test_size=0.01, random_state=42)
        
        train_df = merged_df[merged_df.id.isin(train_ids)]
        test_df = merged_df[merged_df.id.isin(test_ids)]
        
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
      
        ID_COLUMN = 'id'
        LABEL_COLUMN = 'category'
        TS_COLUMN = 'ts'
        
        train_df = train_df.drop(segment_start_name, axis=1)
        test_df = test_df.drop(segment_start_name, axis=1)
        
        my_preparer = TensorflowTsDataPreparer(
            id_column=ID_COLUMN,
            label_column=LABEL_COLUMN,
            ts_column=TS_COLUMN, )

        my_preparer.fit(train_df)
        
        train_X, train_y = my_preparer.transform(train_df)
        test_X, test_y = my_preparer.transform(test_df)
        
        train_X, train_y = sampling(train_X, train_y)
        
        num_classes = train_y.shape[1]

        models = modelgen.generate_models(train_X.shape,
                                  number_of_classes=num_classes,
                                  number_of_models = 5, model_type='DeepConvLSTM'
                                 )
        
        histories, val_accuracies, val_losses = train_models_on_samples(train_X, train_y,
                                                                           test_X, test_y,
                                                                           models,nr_epochs=30, batch_size=avg_segment,
                                                                           subset_size=len(train_X),
                                                                           verbose=True, early_stopping=True)
        
        best_model_index = np.argmax(val_accuracies)
        best_model, best_params, best_model_types = models_LSTM[best_model_index]

        
        self.logger.info('Best model: %s', best_model_types)
        self.logger.info('Best params: %s', best_params)
        
        lst = [df_segments,  df_telemetry, labeled_segment_df, merged_df, train_df, test_df]
        
        del df_segments, df_telemetry, labeled_segment_df, merged_df, train_df, test_df 

        del lst  

        best_model.save(actual_model)
        self.__s3_service.upload(actual_model, line_id, avg_segment)
        self.logger.info('Finished processing!')


if __name__ == '__main__':
    processor_service = Processor(S3Service(),
                                  SegmentService(),
                                  DataService())

    processor_service.calculate_weights_and_upload()