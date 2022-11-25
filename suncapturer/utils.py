import random
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import torch

DEFAULT_FMT = '%Y-%m-%d %H:%M:%S'


def standardize_time(t):
    if isinstance(t, int):
        t = datetime.fromtimestamp(t)
        t = t.strftime(DEFAULT_FMT)

    else:
        t = t.split('+')[0]
        t = t.replace('T', ' ')

    return t


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Requestor:
    def __init__(self, token):
        self._api_url = 'https://research-api.dershare.xyz'
        self._auth_param = {'headers': {'Authorization': f'Bearer {token}'}}

    def _get(self, url):
        '''
        주어진 url의 리소스를 조회한다.

        Args:
            url (str): API url
        '''
        response = requests.get(url, **self._auth_param)

        return pd.DataFrame(response.json())

    def _post(self, url, data):
        '''
        리소스 생성 데이터를 이용해서 주어진 url의 리소스를 생성한다.

        Args:
            url (str): API url
            data (dict): 리소스 생성용 데이터
        '''
        response = requests.post(url, json=data, **self._auth_param)

        return response.json()

    def get_pv_sites(self):
        '''
        태양광 발전소 목록 조회
        '''
        url = 'open-proc/cmpt-2022/pv-sites'
        pv_sites = self._get(f'{self._api_url}/{url}')

        return pv_sites

    def get_pv_gens(self, date):
        '''
        태양광 발전소별 발전량 조회. 주어진 날짜의 전체 발전소별 발전량을 가져온다
        '''
        url = f'open-proc/cmpt-2022/pv-gens/{date}'
        pv_gens = self._get(f'{self._api_url}/{url}')

        return pv_gens

    def get_weathers(self, date):
        '''
        기상 관측 정보 조회. 주어진 날짜의 3가지 기상데이터별로 별도로 조회해야 하며, 종관기상관측 데이터도 별도로 조회가능한다.
        '''
        weathers = []
        for i in [1, 2, 3]:
            url = f'open-proc/cmpt-2022/weathers/{i}/observeds/{date}'
            weathers.append(self._get(f'{self._api_url}/{url}'))

        return weathers

    def get_forecasts(self, obs_id, date, hour):
        '''
        기상 예측 정보 조회. 주어진 날짜의 특정 시간대에 예측된 기상 예측 정보를 조회할 수 있다. 3가지 기상데이터별로 별도로 조회해야 한다.
        '''
        url = f'open-proc/cmpt-2022/weathers/1/{obs_id}/forecasts/{date}/{hour}'
        fcst = self._get(f'{self._api_url}/{url}')

        return fcst

    def get_environments(self, date):
        '''
        광명발전소의 센서 데이터 조회. 주어진 날짜의 특정 시간대에 측정된 센서 데이터를 조회할 수 있다.
        '''
        url = f'open-proc/cmpt-2022/evironments/{date}'
        environments = self._get(f'{self._api_url}/{url}')

        return environments

    def _post_bids(self, amounts):
        '''
        집합 자원 태양광 발전량 입찰. 시간별 24개의 발전량을 입찰하며 API가 호출된 시간에 따라 입찰 대상일이 결정된다.

        Args:
            amouts: list of 24 dictionaries, each dictionary should have two keys, 'upper' and 'lower'
        '''
        url = 'open-proc/cmpt-2022/bids'
        success = self._post(f'{self._api_url}/{url}', amounts)

        return success
