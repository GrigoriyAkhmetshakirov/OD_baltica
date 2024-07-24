Подготовка:
1. Загрузить и установить Python по ссылке `https://www.python.org/downloads/release/python-380/`;
2. Командой `pip install --upgrade pip` обновить pip;
2. Загрузить и установить Docker по ссылке `https://www.docker.com` под необходимую ОС.


Для аннтотации изображений необходимо выполнить следующие шаги:
1. Перейти в директорию `image_annotation`;
2. В директорию `images` поместить изображения для разметки;
3. Открыть терминал и перейти в директорию `labelImg`;
4. Командой `pip3 install labelImg` установить необходимые пакеты;
5. Командой `labelImg` запустить программу для разметки;
6. Выполнить разметку используя лейблы 'pena','fontan' и 'shapka' аналогично тому ка это сделано на видео https://www.youtube.com/watch?v=Tlvy-eM8YO4&t=370s,




Для обучения модели на новых данных необходимо выполнить следующие шаги:
1. Перейти в директорию, где хранится скрипт;
2. Перейти в дерикторию `./Tensorflow/workspace/images`, в папки `train` и `test` поместить изображения и файлы разметки в формате .xml;
3. Запустить Docker;
4. Перейти в директорию, где хранится скрипт и открыть терминал;
5. Командой `docker-compose build` сконфигурировать контейнер;
6. Открыть новое окно терминала, перейти в директорию, где хранится скрипт и командой `docker exec -it myapp_container bash` зайти в контейнер;
7. Командой `python script_main.py -s=1000 -ve=False -e=False` запустить обучение модели;
8. Командой `exit` выйти из контейнера;
9. Копировать файлы обученной модели командой `docker cp myapp_container:./root/models.tar.gz ./`;
10. В папке со скриптом появится архив с обученной моделью.