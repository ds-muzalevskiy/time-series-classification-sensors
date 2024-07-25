import boto3
import os
import time
import json

import pandas as pd

from .BaseService import BaseService


class S3Service(BaseService):

    def __init__(self):
        BaseService.__init__(self)
        self.__s3 = boto3.resource('s3')
        self.__client = boto3.client('s3')
        self.avg_segment = 0
        self.__created = ''
        self.__tag = ''
        self.__archive_bucket = 'mqtt-metrics-archive'

    def upload(self, actual_model, line_id, avg_segment):
        try:
            bucket = self.__s3.Bucket(self.bucket)
            key = S3Service.get_s3_key(actual_model, line_id)
            bucket.upload_file(actual_model, key, ExtraArgs={
                "Metadata": {"avg_segment": str(avg_segment),
                             "created": str(time.time()),
                             "img_tag": os.environ['VERSION']}})
        except Exception as e:
            self.logger.exception("Failed to upload file", e)

    def to_df(self, line_id, feature):
        prefix = '%s/%s' % (line_id, feature)
        response = self.__client.list_objects(
            Bucket=self.__archive_bucket,
            Prefix=prefix,
            RequestPayer='requester'
        )
        contents = response['Contents']
        res_df = []
        for content in contents:
            file = content['Key']
            obj = self.__s3.Object(self.__archive_bucket, file)
            body = obj.get()['Body'].read()
            json_body = json.loads(body)
            values = json_body['values']
            df = pd.DataFrame(values)
            res_df.append(df) 
            result_df = pd.concat(res_df)
        return result_df

    def download(self, filename, line_id):
        bucket = self.__s3.Bucket(self.bucket)
        key = S3Service.get_s3_key(filename, line_id)
        while os.environ['VERSION'] != self.get_metadata(filename, line_id)['img_tag']:
            self.logger.info('File not available yet, target version is %s. Waiting 60 secs', os.environ['VERSION'])
            time.sleep(60)
        metadata = self.get_metadata(filename, line_id)
        bucket.download_file(key, filename)
        self.__created = str(metadata['created'])
        self.__tag = metadata['img_tag']
        self.avg_segment = metadata['avg_segment']

    def get_metadata(self, filename, line_id):
        key = S3Service.get_s3_key(filename, line_id)
        return self.__client.head_object(Bucket=self.bucket, Key=key)['Metadata']

    @staticmethod
    def get_s3_key(filename, line_id):
        return line_id + "__" + filename

    def is_file_up_to_date(self, filename, line_id):
        if not(os.path.isfile(filename)):
            self.logger.info('Local file not found')
            return False
        metadata = self.get_metadata(filename, line_id)
        return metadata['created'] == self.__created and metadata['img_tag'] == self.__tag
