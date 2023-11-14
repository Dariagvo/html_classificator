from bs4 import BeautifulSoup
import requests
import csv


class Parser:
    def return_url_info(self, url: str) -> str:
        url = "https://" + url
        result = ""

        # Проверяет, что можно через requests.get(url)
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            return ""

        # рабочая страница
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

        return result

    def parse(self, table: str):
        with open(table, 'r') as file:
            csv_reader = csv.reader(file, delimiter=';')
            next(csv_reader)

            for row in csv_reader:
                url = row[0]
                base_category_nm = row[1]

                print(f"URL: {url}, Категория: {base_category_nm}")
                result = self.return_url_info(str(url))

                if result == "":
                    continue

                print(result)
                break
