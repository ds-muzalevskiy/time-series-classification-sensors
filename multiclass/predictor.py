import warnings
warnings.filterwarnings('ignore')

import os
import datetime
import pandas as pd
import numpy as np
import traceback
from keras.models import load_model
from lib.data_preparation import TensorflowTsDataPreparer
from lib.merge_all_in_tolerance import merge_asof_outer

from services.S3Service import S3Service
from services.SqsListener import SqsListener
from services.DataService import DataService
from services.SuggestionService import SuggestionService

actual_model = 'actual_model.h5'


class CompletedSegmentListener(SqsListener):

    def __init__(self, queue, **kwargs):
        super().__init__(queue, **kwargs)
        self.__s3 = S3Service()

    def handle_message(self, body):
        try:
            data_service = DataService()
            suggestion_service = SuggestionService()
            line_id = body['lineId']
            segment_id = body['id']
            start_time = datetime.datetime.fromtimestamp(body['startTime'] / 1000.0)
            end_time = datetime.datetime.fromtimestamp(body['endTime'] / 1000.0)
            df_segments = data_service.prepare_segments_with_cards(line_id, start_time, end_time)
            df_telemetry = data_service.prepare_telemetry(line_id, start_time - datetime.timedelta(days=1), end_time)
            if not(self.__s3.is_file_up_to_date(actual_model, line_id)):
                self.logger.info('Weights file not up to date, downloading..')
                self.__s3.download(actual_model, line_id)
            avg_segment = self.__s3.avg_segment
            avg_segment = int(avg_segment)
            loaded_model = load_model(actual_model)

            telemetry_ts_name = 'Date'
            category_name = 'card.rootCause.id'
            id_name = 'id'
            segment_start_name = 'startTime'

            df_segments[id_name] = segment_id
            df_segments[category_name] = 'val'
            df_segments = df_segments[[id_name, segment_start_name, category_name]]

            time_before_downtime = pd.to_timedelta(avg_segment, unit='m')

            merged_df = merge_asof_outer(df_segments,
                                         df_telemetry,
                                         left_on=segment_start_name,
                                         right_on=telemetry_ts_name,
                                         tolerance=time_before_downtime,
                                         )

            merged_df = merged_df.rename(columns={category_name: 'category', telemetry_ts_name: 'ts', })
          
            ts_lengths_by_id = merged_df.groupby(id_name).size()

            num_measurements = avg_segment
            all_ids = ts_lengths_by_id[ts_lengths_by_id == num_measurements].index.values

            merged_df = merged_df[merged_df.id.isin(all_ids)]
            
            ID_COLUMN = 'id'
            LABEL_COLUMN = 'category'
            TS_COLUMN = 'ts'

            merged_df = merged_df.drop(segment_start_name, axis=1)
            merged_df.category = 'val'

            my_preparer = TensorflowTsDataPreparer(
                id_column=ID_COLUMN,
                label_column=LABEL_COLUMN,
                ts_column=TS_COLUMN, )

            my_preparer.fit(merged_df)

            X, y = my_preparer.transform(merged_df)

            merged_df = merged_df.drop_duplicates(ID_COLUMN, keep='first')
            merged_df = merged_df.reset_index(drop=True)

            y_pred = loaded_model.predict(X)
            df_pred = pd.DataFrame(np.round(y_pred))
            df_pred = df_pred.rename(
                columns={df_pred.columns[0]: 'line_id', df_pred.columns[1]: 'other'})
            df_pred['Pred'] = df_pred['line_id'].apply(
                lambda x: 'line_id' if x == 1 else 'other')
            id_val = 'id'
            pred_val = 'Pred'
            df = pd.concat([merged_df, df_pred], axis=1)
            res_df = df[[id_val, pred_val]]

            suggestion = res_df.iloc[0]['Pred']
            if suggestion is not None and suggestion != 'other':
                self.logger.info('Sending suggestion %s', suggestion)
                suggestion_service.post_suggestions(segment_id, suggestion)
        except (Exception, ValueError, TypeError):
            self.logger.error('Failed to process %s', body)
            self.logger.error(traceback.format_exc())


listener = CompletedSegmentListener(os.environ.get('SOURCE_QUEUE'), force_delete=True)
listener.listen()
