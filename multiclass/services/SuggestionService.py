import requests
import json

from .BaseService import BaseService
from .LoginService import LoginService


class SuggestionService(BaseService):
    def __init__(self):
        BaseService.__init__(self)
        self.__login_service = LoginService()

    def post_suggestions(self, segment_id, suggestion):
        headers = self.__login_service.get_login_headers()
        url = "https://url.execute-api.eu-central-1.amazonaws.com/stage/api/suggestions"
        root_cause = {'rootCause': suggestion}
        data = {'id': segment_id, 'suggestion': root_cause}
        json_data = json.dumps(data)
        try:
            requests.post(url, headers=headers, data=json_data)
        except Exception as e:
            self.logger.exception("Failed to post suggestions", e)
