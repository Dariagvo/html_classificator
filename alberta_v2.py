#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().system('pip install wandb')


# In[15]:


import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.notebook import tqdm
import wandb
import time
import random
import matplotlib.pyplot as plt


# In[16]:


# Создаем список для итерации
items = list(range(10))

# Используем tqdm для создания прогресс-бара
for item in tqdm(items, desc='Прогресс'):
    # Имитируем задержку для наглядности
    time.sleep(0.5)


# In[18]:


wandb.login()
wandb.init(project="albert01")

# Увеличиваем частоту логирования в wandb
log_interval = 100

# Изменяем число эпох
epochs = 5

# Логируем градиенты
wandb.config.log_gradients = True

# Логируем метрики на валидации перед обучением на общем графике
wandb.config.log_validation_metrics = True


# In[19]:


data = pd.read_csv('/kaggle/input/zxcvbn/pars_data.csv', delimiter=';')
data.columns = ['text', 'target']
data.dropna(inplace=True)
class_counts = data['target'].value_counts()
print("Количество примеров в каждом классе:")
print(class_counts)
num_samples = len(data)
num_classes = data['target'].nunique()

print("\nОбщее количество образцов:", num_samples)
print("Количество классов:", num_classes)

print("\nПервые 100 символов из 10 рандомных текстов:")
for _ in range(10):
    random_index = random.randint(0, len(data) - 1)
    text = data['text'][random_index]
    print(f'{_ + 1}. {text[:100]}')

text_lengths = data['text'].apply(lambda x: len(x.split()))
print("\nРаспределение длин текстов:")
print("Средняя длина текста:", text_lengths.mean())
print("Минимальная длина текста:", text_lengths.min())
print("Максимальная длина текста:", text_lengths.max())
print("Медианная длина текста:", text_lengths.median())
print("Стандартное отклонение длины текста:", text_lengths.std())

max_length_limit = 5000
filtered_lengths = text_lengths[text_lengths < max_length_limit]
plt.figure(figsize=(10, 6))
plt.hist(filtered_lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Распределение длин строк без выбросов', fontsize=16)
plt.xlabel('Длина строки', fontsize=14)
plt.ylabel('Частота', fontsize=14)
plt.grid(True)
plt.show()


# In[20]:


def tokenize_text(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Tokenization Progress"):
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['target'], test_size=0.2,
                                                                      random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5,
                                                                  random_state=42)

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

max_length = 512

train_input_ids, train_attention_masks = tokenize_text(train_texts, tokenizer, max_length)
val_input_ids, val_attention_masks = tokenize_text(val_texts, tokenizer, max_length)
test_input_ids, test_attention_masks = tokenize_text(test_texts, tokenizer, max_length)


# In[21]:


batch_size = 16

train_dataset = TensorDataset(train_input_ids, train_attention_masks, torch.tensor(train_labels.values))
val_dataset = TensorDataset(val_input_ids, val_attention_masks, torch.tensor(val_labels.values))
test_dataset = TensorDataset(test_input_ids, test_attention_masks, torch.tensor(test_labels.values))

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


# In[22]:


model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
model.cuda()
optimizer = AdamW(model.parameters(), lr=1e-5)

class_weights = torch.tensor([class_counts[1] / class_counts[0], 1.0], dtype=torch.float).cuda()
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)


# In[23]:


# Логируем градиенты
wandb.watch(model, loss_function, log="all")


# In[ ]:


for epoch in range(epochs):
    model.train()
    train_losses = []

    for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch: {}".format(epoch + 1))):
        batch = tuple(t.cuda() for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_function(outputs.logits, inputs['labels'])
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Логируем обучение каждые 100 шагов
        if (step + 1) % 100 == 0: 
            wandb.log({"Train Loss": loss.item()})

    train_loss = np.mean(train_losses)

    model.eval()
    val_losses = []
    val_predictions = []
    val_true_labels = []

    for batch in val_dataloader:
        batch = tuple(t.cuda() for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        with torch.no_grad():
            outputs = model(**inputs)

        loss = loss_function(outputs.logits, inputs['labels'])
        val_losses.append(loss.item())
        logits = outputs.logits
        val_predictions.extend(torch.argmax(logits, dim=1).tolist())
        val_true_labels.extend(inputs['labels'].tolist())

    val_loss = np.mean(val_losses)
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_precision = precision_score(val_true_labels, val_predictions)
    val_recall = recall_score(val_true_labels, val_predictions)
    val_f1 = f1_score(val_true_labels, val_predictions)

    wandb.log({"Val Loss": val_loss,
               "Val Accuracy": val_accuracy,
               "Val Precision": val_precision,
               "Val Recall": val_recall,
               "Val F1": val_f1})

    print("Epoch {} - Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}, Val Precision: {:.4f}, "
          "Val Recall: {:.4f}, Val F1: {:.4f}".format(
        epoch + 1, train_loss, val_loss, val_accuracy, val_precision, val_recall, val_f1))


# In[13]:


test_predictions = []
test_true_labels = []

for batch in test_dataloader:
    batch = tuple(t.cuda() for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    test_predictions.extend(torch.argmax(logits, dim=1).tolist())
    test_true_labels.extend(inputs['labels'].tolist())

test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_precision = precision_score(test_true_labels, test_predictions)
test_recall = recall_score(test_true_labels, test_predictions)
test_f1 = f1_score(test_true_labels, test_predictions)

print("Test Accuracy: {:.4f}, Test Precision: {:.4f}, Test Recall: {:.4f}, Test F1: {:.4f}".format(
    test_accuracy, test_precision, test_recall, test_f1))

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_true_labels, test_predictions)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]

print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

