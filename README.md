### Домашнее задание по курсу MLOps
#### Проект предсказание цены на недвижимость в Калифорнии, США
Датасет представляет из медианные параметры недвижимости по районам, целевая переменная -- цена недвижимости.
Предсказание делается на основе параметров:
 - Медианный доход жильцов
 - Медианный возраст жильцов
 - Среднее количество комнат в квартире
 - Среднее количество спален в квартире
 - Количество жильцов
 - Координаты дома

Ссылка на информацию про датасет: [link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

#### MLOps часть
1. Настроена конфигурация через hydra
2. Файлы датасета поставляются через dvc
3. Настроены прекоммитные хуки через pre-commit
4. Настроена поставка зависимостей через poetry
5. Добавлено логгирование через mlflow 
