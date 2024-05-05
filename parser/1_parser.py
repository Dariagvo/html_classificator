import csv
import requests as re
from bs4 import BeautifulSoup
import re as regex
import time

total_sites = sum(1 for line in open('hse_x_td_url.csv', 'r', encoding='utf-8'))

start_time = time.time()  # время начала работы скрипта

with open('parsed_data.csv', 'w', encoding='utf-8', newline='') as output_file:
    # заголовки для новой таблицы
    fieldnames = ['url', 'base_category_nm', 'parsed_text']
    # писатель для записи в CSV
    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    # заголовки в файл
    csv_writer.writeheader()

    with open('hse_x_td_url.csv', 'r', encoding='utf-8') as file:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36 '
        }

        csv_reader = csv.reader(file, delimiter=';')

        next(csv_reader)

        for index, row in enumerate(csv_reader, start=1):  # для нумерации сайтов
            url = "https://" + row[0]
            base_category_nm = row[1]

            # проверка, на необрабатываемые нами сайты
            social_networks = ['vk.com', 'facebook.com', 'instagram.com', 't.me', 'wa.me']
            is_social_network = any(domain in url for domain in social_networks)

            if is_social_network:
                print(f"Сайт {url} является социальной сетью, пропускаем.")
                continue

            if 'getcourse.ru' in url:
                print(f"Сайт {url} содержит getcourse.ru, пропускаем.")
                continue
            if 'taplink.cc' in url or 'taplink.ws' in url:
                print(f"Сайт {url} содержит taplink, пропускаем.")
                continue

            print(f"--> Сайт {index} / {total_sites} в датасете. URL: {url}, Категория: {base_category_nm}")

            try:
                response = re.get(url, headers=headers, timeout=10)
                response.encoding = response.apparent_encoding  # Определение правильной кодировки

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # description
                    meta_description = soup.find('meta', attrs={'name': 'description'})

                    # получаем его содержимое, если есть
                    description_content = meta_description[
                        'content'] if meta_description and 'content' in meta_description.attrs else ''

                    # получаем весь текст со страницы
                    all_text = ' '.join([element.get_text(strip=True) for element in soup.find_all(string=True)])

                    all_text_with_description = f"{description_content} {all_text}"

                    # удаляем ссылки
                    all_text_with_description = regex.sub(r'http\S+', '', all_text_with_description)

                    # заменяем символы, которые не являются буквами или цифрами на пробел
                    all_text_with_description = ''.join([char if char.isalpha() or char.isdigit() or char.isspace() else ' ' for char in all_text_with_description])

                    # удаляем лишние пробелы
                    all_text_with_description = ' '.join(all_text_with_description.split())

                    print(f"Полный текст: {all_text_with_description}")

                    # проверка на длину текста
                    if len(all_text_with_description) < 50:
                        print("Текст слишком короткий, пропускаем.")
                        continue

                    # исключение строк, содержащих в текстах некоторые подстроки
                    exclude_substrings = ['Sorry your request has been denied', 'Your account is suspended', 'without JavaScript', 
                                        'Сайт заблокирован', 'Действие аккаунта приостановлено', 'проводятся технические работы', 
                                        'Сайт временно недоступен']  # список можно пополнять
                    if any(substring in all_text_with_description for substring in exclude_substrings):
                        print("Текст содержит недопустимые подстроки, пропускаем.")
                        continue

                    # удаление предложений из текстов
                    exclude_sentences = ['You need to enable JavaScript to run this app', 'Made on Tilda']  # список можно пополнять
                    for sentence in exclude_sentences:
                        all_text_with_description = all_text_with_description.replace(sentence, '')

                    # данные в новую таблицу
                    csv_writer.writerow({'url': url, 'base_category_nm': base_category_nm, 'parsed_text': all_text_with_description})
                elif response.status_code != 403 and response.status_code != 404:
                    print(f"Ошибка при получении страницы: {response.status_code}.")
                else:
                    print(f"Ошибка при получении страницы: {response.status_code}.")
            except Exception as e:
                # if not isinstance(e, re.ConnectionError):
                print(f"Ошибка при обработке данного сайта.")

            # время, прошедшее с начала работы скрипта
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(
                f"Прошло времени: {elapsed_time:.2f} сек., средняя скорость обработки 1 сайта: {(elapsed_time / index):.2f} сек.")
