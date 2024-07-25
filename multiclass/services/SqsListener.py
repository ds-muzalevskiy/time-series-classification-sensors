import boto3
import time
import traceback
import json
from abc import abstractmethod
from .BaseService import BaseService

line_id = 'id'


class SqsListener(BaseService):

    def __init__(self, queue, **kwargs):
        BaseService.__init__(self)

        self._queue = queue
        self._poll_interval = kwargs['interval'] if 'interval' in kwargs else 1
        self._force_delete = kwargs['force_delete'] if 'force_delete' in kwargs else False
        self._wait_time = kwargs['wait_time'] if 'wait_time' in kwargs else 0
        self._client = self._initialize_client(queue)

    @staticmethod
    def _initialize_client(queue):
        sqs = boto3.resource('sqs', 'eu-central-1')
        return sqs.get_queue_by_name(QueueName=queue)

    def listen(self):
        self.logger.info('Start listening queue %s', self._queue)
        while True:

            messages = self._client.receive_messages(
                WaitTimeSeconds=self._wait_time,
            )

            for message in messages:
                try:
                    params_dict = json.loads(message.body)
                    if SqsListener.filter(params_dict['lineId'], params_dict['level']):
                        self.logger.info('Started processing message %s', message.body)
                        self.handle_message(params_dict)
                except (Exception, ValueError, TypeError) as ex:
                    self.logger.error('Failed to handle %s', message.body)
                    self.logger.error(traceback.format_exc(ex))

                if self._force_delete:
                    message.delete()
            else:
                time.sleep(self._poll_interval)


    @staticmethod
    def filter(line_id, level):
        return line_id == wepa_line_id and level != 2
