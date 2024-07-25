import requests

from .BaseService import BaseService


class LoginService(BaseService):
    def __init__(self):
        BaseService.__init__(self)

    def get_login_headers(self):
        data = {'grant_type': 'client_credentials',
                'scope': 'read write',
                'client_secret': 'secret',
                'client_id': 'id'
                }
        oauth_url = 'https://%s.url.de/api/oauth/token' % self.oauth
        response = requests.post(oauth_url, headers={'Accept': 'application/json',
                                                     'Content-Type': 'application/x-www-form-urlencoded'}, data=data)

        data = response.json()
        authorization = 'Bearer ' + data.get('access_token')
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'accept-language': 'en',
            'Authorization': authorization
        }
        return headers
