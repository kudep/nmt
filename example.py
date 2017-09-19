import nmt.interface as bot
import os


model_path = os.environ.get("CHITCHAT_MODEL")
bot.init(model_path)


bot.interactive_dialogue(prompt = "[Ваш запрос]: ")
