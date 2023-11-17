# извините я не разделил на классы аааааа
import csv
import requests as re
from bs4 import BeautifulSoup

with open('/Users/macbook/Downloads/hse_x_td_url.csv', 'r', encoding='utf-8') as file:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36 '
    }

    csv_reader = csv.reader(file, delimiter=';')

    next(csv_reader)

    for row in csv_reader:
        url = "https://" + row[0]
        base_category_nm = row[1]

        print(f"URL: {url}, Категория: {base_category_nm}")

        try:
            response = re.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # извлекаем description
                    meta_description = soup.find('meta', attrs={'name': 'description'})

                    # получаем его содержимое, если есть
                    description_content = meta_description[
                        'content'] if meta_description and 'content' in meta_description.attrs else ''

                    # получаем весь текст со страницы
                    all_text = ' '.join([element.get_text(strip=True) for element in soup.find_all(string=True)])

                    all_text_with_description = f"{description_content} {all_text}"

                    print(f"All Text with Description: {all_text_with_description}")
            elif response.status_code != 403 and response.status_code != 404:
                print(f"Ошибка при получении страницы: {response.status_code}")
        except Exception as e:
            if not isinstance(e, re.ConnectionError):
                print(f"Ошибка при обработке URL {url}: {type(e).__name__}, {e}")
