#!/bin/python3
# -*- coding: utf-8 -*-
from os.path import join
import json
__all__ = ["get_config"]

CONFIG_DIR = "data/configs"
DB_DIR = "data/db"
BOT_TOKEN = join(CONFIG_DIR, "bot_token.json")


def get_config():
    params = {}
    with open(BOT_TOKEN) as jsonf:
        bot_token = json.load(jsonf)
    params['BOT_TOKEN'] = bot_token['release_token']# release_token/debug_token

    return params
