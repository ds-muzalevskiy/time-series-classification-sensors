import logging
import os


class BaseService:
    def __init__(self):
        self.stage = os.environ['STAGE']
        self.bucket = os.environ['S3_BUCKET']
        if self.stage == 'prod':
            self.oauth = 'oauthservice'
            self.apistage = ''
        elif self.stage == 'pre':
            self.oauth = 'oauthservicestage'
            self.apistage = 'stage'
        else:
            self.oauth = 'oauthservicestage'
            self.apistage = 'stage'
        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
