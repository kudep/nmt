import nmt.interface as bot
import os


model_path = os.environ.get("CHITCHAT_MODEL")
ckpt_name = os.environ.get("CHITCHAT_CKPT") #Optional
bot.init(model_path, ckpt_name) # if ckpt_name is None bot gets last ckpt

#print(bot.send('Здравствуйте , Подскажите , веду бух учёт на Эльбе . Контур . Сохранил платёжные поручения для уплаты страховых платежей , сохранил их в текстовом формате  ( так сохраняет сама программа ), а в банк не могу импортировать . пишет , что в файле нет необходимых реквизитов , хотя года открываешь текстовый файл вся информация присутствует . Спасибо .'))

bot.interactive_dialogue(prompt = "Пользователь: ")
