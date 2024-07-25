import datetime
import requests

from .BaseService import BaseService


class SegmentService(BaseService):
    def __init__(self):
        BaseService.__init__(self)

    def get_segments_till_now(self, line_id, start_date, headers, with_cards=False):
        return self.get_segments_for_range(line_id, start_date, datetime.datetime.now(), headers, with_cards)

    def get_segments_for_range(self, line_id, start_date, end_date, headers, with_cards=False):
        output = dict()
        try:
            url = "https://url.execute-api.eu-central-1.amazonaws.com/%s/%s?start=%d&end=%d"
            if with_cards:
                url = url + "&metric=segments&withCards=true"

            temp_end_date = min(end_date, start_date + datetime.timedelta(days=7))
            finished = False
            while temp_end_date <= end_date and not finished:
                if temp_end_date == end_date:
                    finished = True
                tmp_url = url % (self.stage, line_id, start_date.timestamp() * 1000, temp_end_date.timestamp() * 1000)
                response = requests.get(tmp_url, headers=headers)
                temp_data = response.json()
                for key, value in temp_data.items():
                    output_arr = output.get(key, [])
                    output_arr.extend(value)
                    output[key] = output_arr
                start_date = temp_end_date + datetime.timedelta(seconds=1)
                temp_end_date = min(end_date, temp_end_date + datetime.timedelta(days=7))
        except Exception as e:
            self.logger.error(e)
        return output

    def get_alarms_for_range(self, line_id, start_date, end_date, headers):
        output = dict()
        try:
            url = "https://url.execute-api.eu-central-1.amazonaws.com/%s/%s?start=%d&end=%d&metric=alarms"

            temp_end_date = min(end_date, start_date + datetime.timedelta(days=7))
            finished = False
            while temp_end_date <= end_date and not finished:
                if temp_end_date == end_date:
                    finished = True
                tmp_url = url % (self.stage, line_id, start_date.timestamp() * 1000, temp_end_date.timestamp() * 1000)
                response = requests.get(tmp_url, headers=headers)
                temp_data = response.json()
                for key, value in temp_data.items():
                    output_arr = output.get(key, [])
                    output_arr.extend(value)
                    output[key] = output_arr
                start_date = temp_end_date + datetime.timedelta(seconds=1)
                temp_end_date = min(end_date, temp_end_date + datetime.timedelta(days=7))
        except Exception as e:
            self.logger.error(e)
        return output