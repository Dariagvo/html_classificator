from bs4 import BeautifulSoup
import requests as re
import csv


class Parser:
    @staticmethod
    def return_url_info(url: str) -> str | None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36 '
        }

        url = "https://" + url

        # Проверяет, что можно через requests.get(url)
        try:
            response = re.get(url, headers=headers, timeout=10)

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
            if not isinstance(e, re.ConnectionError):
                print(f"Ошибка при обработке URL {url}: {type(e).__name__}, {e}")

    def parse_from_table(self, table: str) -> None:
        with open(table, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=';')
            next(csv_reader)

            for row in csv_reader:
                url = row[0]
                base_category_nm = row[1]

                print(f"URL: {url}, Категория: {base_category_nm}")

                result = self.return_url_info(str(url))

                if result is None:
                    continue

                print("All Text with Description:", result)
