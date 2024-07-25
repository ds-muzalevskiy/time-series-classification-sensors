#!/usr/bin/env python
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from services.S3Service import S3Service
from services.SegmentService import SegmentService
from services.DataService import DataService
from Processor import Processor


service = Processor(S3Service(), SegmentService(), DataService())
service.calculate_weights_and_upload
scheduler = BlockingScheduler()
scheduler.daemonic = False
scheduler.add_job(service.calculate_weights_and_upload, CronTrigger.from_crontab('0 0 * * *'))
scheduler.start()
