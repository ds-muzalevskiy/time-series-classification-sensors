import pandas as pd
from pandas.io.json import json_normalize

from .LoginService import LoginService
from .SegmentService import SegmentService
from .BaseService import BaseService


class DataService(BaseService):
    def __init__(self):
        BaseService.__init__(self)
        self.__login_service = LoginService()
        self.__segment_service = SegmentService()

    def prepare_telemetry(self, line_id, start_date, end_date):
        headers = self.__login_service.get_login_headers()
        data = self.__segment_service.get_segments_for_range(line_id, start_date, end_date, headers)
       
        
        df_log_accu_percentage = pd.DataFrame(data['log_accu_percentage'], columns=['Date', 'log_accu_percentage'])
        df_log_per_minute = pd.DataFrame(data['log_per_minute'], columns=['Date', 'log_per_minute'])
        df_log_produced2 = pd.DataFrame(data['logproduced2'], columns=['Date', 'logproduced2'])
        df_log_rejected2 = pd.DataFrame(data['logrejected2'], columns=['Date', 'logrejected2'])
        df_speed = pd.DataFrame(data['speed'], columns=['Date', 'speed'])
        df_num_stops = pd.DataFrame(data['numstops'], columns=['Date', 'numstops'])
        df_tension = pd.DataFrame(data['tension'], columns=['Date', 'tension'])
        df_tension_eb1 = pd.DataFrame(data['tension_eb1'], columns=['Date', 'tension_eb1'])
        df_tension_eb2 = pd.DataFrame(data['tension_eb2'], columns=['Date', 'tension_eb2'])
        df_tension_rw = pd.DataFrame(data['tension_rw'], columns=['Date', 'tension_rw'])
        df_tension_uw1 = pd.DataFrame(data['tension_uw1'], columns=['Date', 'tension_uw1'])
        df_tension_uw2 = pd.DataFrame(data['tension_uw2'], columns=['Date', 'tension_uw2'])
        df_tension_uw3 = pd.DataFrame(data['tension_uw3'], columns=['Date', 'tension_uw3'])

        
        df_telemetry = df_log_accu_percentage.merge(df_log_per_minute, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_log_produced2, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_log_rejected2, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_num_stops, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_speed, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension_eb1, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension_eb2, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension_rw, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension_uw1, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension_uw2, left_on='Date', right_on='Date', how='inner')
        df_telemetry = df_telemetry.merge(df_tension_uw3, left_on='Date', right_on='Date', how='inner')
        
               
        df_telemetry['Date'] = pd.to_datetime(df_telemetry['Date'], unit='ms')
        return df_telemetry

    def prepare_alarms(self, line_id, start_date, end_date):
        headers = self.__login_service.get_login_headers()
        data = self.__segment_service.get_alarms_for_range(line_id, start_date, end_date, headers)

        df_alarms = json_normalize(data['alarms'])
        df_alarms['time'] = pd.to_datetime(df_alarms.time, unit='ms')
        df_alarms['time'] = df_alarms.time.astype('datetime64[s]')
        df_alarms['time'] = df_alarms.time.map(lambda x: x.replace(second=0))

        ts_column = 'time'
        sev_column = 'severity'

        df_alarms = df_alarms[[sev_column, ts_column]]

        return df_alarms

    def prepare_segments_with_cards(self, line_id, start_time, end_time):
        headers = self.__login_service.get_login_headers()
        data = self.__segment_service.get_segments_for_range(line_id, start_time, end_time, headers, with_cards=True)

        df_segments = json_normalize(data['segments'])

        df_segments['startTime'] = pd.to_datetime(df_segments['startTime'], unit='ms')
        df_segments['endTime'] = pd.to_datetime(df_segments['endTime'], unit='ms')
        return df_segments
