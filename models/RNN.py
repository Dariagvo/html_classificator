# Confusion Matrix:
# [[6295  608]
#  [ 486 2131]]

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.93      0.91      0.92      6903
#            1       0.78      0.81      0.80      2617

#     accuracy                           0.89      9520
#    macro avg       0.85      0.86      0.86      9520
# weighted avg       0.89      0.89      0.89      9520

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# работа с GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# загружаем данные
urls_data = pd.read_csv('urls.csv', on_bad_lines='skip', sep=';', lineterminator='\n')
pars_data = pd.read_csv('pars_data.csv', on_bad_lines='skip', sep=';', lineterminator='\n')

# объединяем
merged_data = pd.merge(urls_data, pars_data, left_index=True, right_index=True)

# предобрабатываем
le = LabelEncoder()
merged_data['target'] = le.fit_transform(merged_data['target'])

train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# удаляем пропущенные значения
train_data = train_data.dropna(subset=['text'])
test_data = test_data.dropna(subset=['text'])

# токенизируем
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['text'])

# переводим в числовые последовательности
X_train = tokenizer.texts_to_sequences(train_data['text'])
X_test = tokenizer.texts_to_sequences(test_data['text'])

# выравниваем по длине
max_length = 200
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# RNN
embedding_dim = 50
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# обучаем
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, train_data['target'],
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1  # вывод прогресса
)

# # оцениваем итог на тесте
# loss, accuracy = model.evaluate(X_test, test_data['target'])
# print(f'Test accuracy: {accuracy * 100:.2f}%')

# получаем предсказания на тесте
predictions = model.predict(X_test)
rounded_predictions = [round(pred[0]) for pred in predictions]

# матрица ошибок и отчет классификации
conf_matrix = confusion_matrix(test_data['target'], rounded_predictions)
class_report = classification_report(test_data['target'], rounded_predictions)

# выводим результат
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nClassification Report:\n{class_report}")
