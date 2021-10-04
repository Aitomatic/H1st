from typing import Optional

import requests


class H1stClient:
    def __init__(self,
                 h1st_ai_server_url: str = 'http://localhost:8000',
                 auth: Optional[str] = None):
        self.h1st_ai_server_url = h1st_ai_server_url
        self.auth = auth

    def list_models(self):
        return requests.get(url=f'{self.h1st_ai_server_url}/h1st/models',
                            headers={'Authorization': self.auth}).json()

    def get_decision(self,
                     model_name_or_uuid: str,
                     data: Optional[dict] = None,
                     files: Optional[dict] = None):
        files_list = []
        if files:
            for k, v in files.items():
                if isinstance(v, (list, tuple)):
                    files_list.extend([(k, open(i, 'rb')) for i in v])
                else:
                    files_list.append((k, open(v, 'rb')))

        return requests.post(url=(f'{self.h1st_ai_server_url}/h1st/models/'
                                  f'{model_name_or_uuid}/exec/'),
                             headers={'Authorization': self.auth},
                             data={} if data is None else data,
                             files=files_list).json()
