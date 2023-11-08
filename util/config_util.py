# -*- coding:utf-8 -*-
import socket
import os
import configparser
from util import path_util


def _get_server_ip():
    return socket.gethostbyname(socket.getfqdn())


# config_{ip}.ini 를 먼저 찾고 없으면 config.ini를 찾도록 하자.
def _get_config_path():
    config_path = os.path.join(
        path_util.get_project_root_path(),
        'config{}config_{}.ini'.format(os.sep, _get_server_ip())
    )
    if not os.path.exists(config_path):
        config_path = os.path.join(
            path_util.get_project_root_path(),
            'config/config.ini'
        )
    return config_path


class ConfigClsf():
    def __init__(self):
        pass

    def get_config(self):
        config = configparser.ConfigParser()
        config_path = _get_config_path()
        config.read(config_path, encoding='utf-8')
        return config


if __name__ == '__main__':
    print(_get_config_path())

