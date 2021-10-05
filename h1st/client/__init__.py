from os import PathLike
from tempfile import NamedTemporaryFile
from typing import Optional, Union

import requests
from streamlit.uploaded_file_manager import UploadedFile


class H1stClient:
    def __init__(self,
                 h1st_ai_server_url: str = 'http://localhost:8000',
                 auth: Optional[str] = None):
        self.h1st_ai_server_url = h1st_ai_server_url
        self.auth = auth

    def list_models(self):
        return requests.get(url=f'{self.h1st_ai_server_url}/h1st/models',
                            headers={'Authorization': self.auth}).json()

    @staticmethod
    def _prep_file_to_upload(file: Union[str, bytes, PathLike, UploadedFile]):
        if isinstance(file, UploadedFile):
            temp_file = NamedTemporaryFile(mode='w+b',
                                           buffering=-1,
                                           encoding=None,
                                           newline=None,
                                           suffix=None,
                                           prefix=None,
                                           dir=None,
                                           delete=True,
                                           errors=None)
            temp_file.write(file.getvalue())

            file = temp_file.name

        return open(file, 'rb')

    def get_decision(self,
                     model_name_or_uuid: str,
                     data: Optional[dict] = None,
                     files: Optional[dict] = None):
        files_list = []
        if files:
            for k, v in files.items():
                if isinstance(v, (list, tuple)):
                    files_list.extend([(k, self._prep_file_to_upload(i))
                                       for i in v])
                else:
                    files_list.append((k, self._prep_file_to_upload(v)))

        return requests.post(url=(f'{self.h1st_ai_server_url}/h1st/models/'
                                  f'{model_name_or_uuid}/exec/'),
                             headers={'Authorization': self.auth},
                             data={} if data is None else data,
                             files=files_list).json()
