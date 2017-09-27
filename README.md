# Neural Machine Translation (seq2seq) for chit-chat

#### Model

```
wget http://share.ipavlov.mipt.ru:8080/repository/models/chitchat/nmt-chitchat-v0.3.1.tgz
export CHITCHAT_MODEL=<PATH_TO_MODEL>
export CHITCHAT_CKPT=<PATH_TO_CKPT> #Optional
```

#### Example
```
from nmt.interface import Agents
import os


model_path = os.environ.get("CHITCHAT_MODEL")
ckpt_name = os.environ.get("CHITCHAT_CKPT") #Optional, it can be not defined

agents = Agents(model_path, ckpt_name = ckpt_name) # if ckpt_name is None bot gets last ckpt

agent_id = '0' # Id for new agent
agents.deploy_agent(id = agent_id, reset = True) # if reset is True and agent is exist then agent context will be reseted. Reset is True by default.
msg = 'Здравствуйте , Подскажите , веду  бух учёт на Эльбе . Контур . Сохранил платёжные поручения для уплаты страховых платежей , сохранил их в текстовом формате  ( так сохраняет сама программа ), а в банк не могу импортировать . пишет , что в файле нет необходимых реквизитов , хотя года открываешь текстовый файл вся информация присутствует . Спасибо .'
print(agents.send(msg,agent_id)) # Id is 0 by default. If agent is not exist then it will be created.


```

#### Example of dialogues

> Ваш запрос:    Добрый день, меня зовут Денис. Мне нужна помощь с новым дизайном Сбербанк Онлайн    

>  done, num sentences 1, time 0s, Tue Sep 19 15:44:34 2017.

> NMT:    Служба технической поддержки Сбербанка , меня зовут Олег , здравствуйте ! Денис , перейдите , пожалуйста , в раздел " Платежи и переводы " , далее " Перейти к списку платежей "


> Ваш запрос:    Спасибо

>  done, num sentences 1, time 0s, Tue Sep 19 15:44:47 2017.

> NMT:    Денис , уточните , пожалуйста , какие затруднения у Вас появляются с заполнением ?


> Ваш запрос:    Все решено

>  done, num sentences 1, time 0s, Tue Sep 19 15:44:56 2017.

> NMT:    Какие затруднения у Вас появляются ?


> Ваш запрос:    Больше никаких, спасибо

>  done, num sentences 1, time 0s, Tue Sep 19 15:45:07 2017.

> NMT:    Вы скопировали платёжное поручение ?


> Ваш запрос:    Да

>  done, num sentences 1, time 0s, Tue Sep 19 15:45:10 2017.

> NMT:    Уточните , пожалуйста , Вы внесли иную строку по поручению ?


> Ваш запрос:    Да

>  done, num sentences 1, time 0s, Tue Sep 19 15:45:16 2017.

> NMT:    Работаете в новом дизайне ?


> Ваш запрос:    Да

>  done, num sentences 1, time 0s, Tue Sep 19 15:45:19 2017.

> NMT:    Данные профиля необходимы для возможности самостоятельной смены пароля на вход в систему в случае его утери . Данные профиля Вы можете заполнить , нажав на стрелку вниз в правом верхнем углу ( рядом с наименованием организации ) и перейдя в раздел " Мой профиль " . Возможно , у Вас остались еще какие-либо вопросы ко мне ? Могу я еще


> Ваш запрос:    Нет

>  done, num sentences 1, time 0s, Tue Sep 19 15:45:30 2017.

> NMT:    Если у Вас ещё возникнут вопросы по системе Сбербанк Бизнес Онлайн , будем рады Вам помочь . Спасибо за обращение , всего доброго !
