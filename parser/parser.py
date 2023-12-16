from bs4 import BeautifulSoup
import requests as req
import re
import string
import csv
import pandas as pd
import typing as tp


class Parser:
    _pos_category = "Образование"

    @staticmethod
    def return_url_info(url: str) -> str | None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36 '
        }

        url = "https://" + url

        # Проверяет, что можно через requests.get(url)
        try:
            response = req.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # извлекаем description
                meta_description = soup.find('meta',
                                             attrs={'name': 'description'})

                # получаем его содержимое, если есть
                description_content = meta_description[
                    'content'] if meta_description and 'content' in meta_description.attrs else ''

                # получаем весь текст со страницы
                all_text = ' '.join(
                    [element.get_text(strip=True) for element in
                     soup.find_all(string=True)])

                all_text_with_description = f"{description_content} {all_text}"
                return all_text_with_description
            elif response.status_code != 403 and response.status_code != 404:
                print(f"Ошибка при получении страницы: {response.status_code}")
        except Exception as e:
            if not isinstance(e, req.ConnectionError):
                print(f"Ошибка при обработке URL {url}: {type(e).__name__}, {e}")

    def parse_from_table(self, table: str, head: int | None = None, show: bool = True, return_data: bool = False) -> tp.Any:
        with open(table, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=';')
            next(csv_reader)

            for ind, row in enumerate(csv_reader):
                if head and ind == head:
                    return

                url = row[0]
                base_category_nm = row[1]

                result = self.return_url_info(str(url))

                if result is None:
                    continue

                if show:
                    print(f"URL: {url}, Категория: {base_category_nm}")
                    print("All Text with Description:", result)
                if return_data:
                    yield result, 1 if base_category_nm == self._pos_category else 0

    def set_pos_category(self, new_cat: str) -> None:
        self._pos_category = new_cat

    @staticmethod
    def value_count(table: str, column: str = 'base_category_nm') -> None:
        data = pd.read_csv(table, on_bad_lines='skip', sep=';', lineterminator='\n')
        print(data[column].value_counts())

    @staticmethod
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_url(text):
        url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub(r'', text)

    @staticmethod
    def remove_punctuation(text):
        delete_dict = {sp_character: '' for sp_character in string.punctuation}
        delete_dict[' '] = ' '
        table = str.maketrans(delete_dict)
        text1 = text.translate(table)
        textArr = text1.split()
        text2 = ' '.join([w for w in textArr if (
                    not w.isdigit() and (not w.isdigit() and len(w) > 2))])

        return text2.lower()

    @staticmethod
    def clean_text(text):
        result = Parser.remove_emoji(text)
        result = Parser.remove_url(result)
        result = Parser.remove_punctuation(result)
        return result
