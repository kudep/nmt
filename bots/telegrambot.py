import configurations
import telebot
from telebot import types
import os
import sys
import threading


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from nmt.interface import Agents

params = configurations.get_config()

bot = telebot.TeleBot(params['BOT_TOKEN'])
print(params['BOT_TOKEN'])
model_path = os.environ.get("CHITCHAT_MODEL")
ckpt_name = os.environ.get("CHITCHAT_CKPT") #Optional, it can be not defined
embeddings_model = os.environ.get("CHITCHAT_EMBED_MODEL") #Optional, it can be not defined

bpe_enable = os.environ.get("CHITCHAT_BPE_ENABLE") #Optional, it can be not defined

bpe_enc_enable = os.environ.get("CHITCHAT_BPE_ENC_ENABLE") #Optional, it can be not defined
bpe_dec_enable = os.environ.get("CHITCHAT_BPE_DEC_ENABLE") #Optional, it can be not defined

bpe_enc_enable = bpe_enable or bpe_enc_enable
bpe_dec_enable = bpe_enable or bpe_dec_enable

agents = Agents(model_path, ckpt_name = ckpt_name,
    embedding_generator_path=embeddings_model, context_len=20, man_context_len=10, bpe_enc=bpe_enc_enable, bpe_dec=bpe_dec_enable)


@bot.message_handler(commands=['start'])
def start_worker(message):
    user_id = message.from_user.id
    agents.reset_agent(user_id)
    bot.send_message(user_id, 'Вы можешете начать диалог')


@bot.message_handler(commands=['reset'])
def reset_agent_worker(message):
    user_id = message.from_user.id
    agents.reset_agent(user_id)
    bot.send_message(user_id, 'Контекст диалога сброшен')


def thread_story(user_id):
    for _ in range(10):
        answer = agents.send(' ', user_id)
        bot.send_message(user_id, answer)


@bot.message_handler(commands=['story'])
def story_agent_worker(message):
    user_id = message.from_user.id
    text = message.text[len('/story'):]

    bot.send_message(user_id, 'Ваш запрос: ' + text)
    answer = agents.send(text, user_id)
    bot.send_message(user_id, answer)

    thread = threading.Thread(target=thread_story, args=(user_id,))
    thread.start()


@bot.message_handler(content_types=["text"])
def default_agent_worker(message):
    user_id = message.from_user.id
    text = message.text
    answer = agents.send(text, user_id)
    bot.send_message(user_id, answer)


if __name__ == "__main__":
    bot.polling(none_stop=True)
