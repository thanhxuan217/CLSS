import os
import re

files = ['d:/WorkSpace/CLSS/flair/trainers/finetune_trainer.py', 'd:/WorkSpace/CLSS/flair/trainers/distillation_trainer.py']

for f in files:
    with open(f, 'r', encoding='utf8') as file:
        content = file.read()
    
    # Replace ColumnDataLoader(list(xyz), ... ) with LazyColumnDataLoader(xyz, ...)
    content = re.sub(r'ColumnDataLoader\(list\(([^)]+)\),', r'LazyColumnDataLoader(\1,', content)
    
    # Replace ColumnDataLoader(ConcatDataset(xyz), ...)
    content = content.replace('ColumnDataLoader(ConcatDataset(', 'LazyColumnDataLoader(ConcatDataset(')
    
    # Replace ColumnDataLoader(xyz, ...)
    content = content.replace('ColumnDataLoader(train_data,', 'LazyColumnDataLoader(train_data,')
    content = content.replace('ColumnDataLoader(dev_data,', 'LazyColumnDataLoader(dev_data,')
    content = content.replace('ColumnDataLoader(self.unlabeled_corpus,', 'LazyColumnDataLoader(self.unlabeled_corpus,')
    
    # Import replacement
    content = content.replace('from ..custom_data_loader import ColumnDataLoader', 'from ..custom_data_loader import ColumnDataLoader, LazyColumnDataLoader')
    
    with open(f, 'w', encoding='utf8') as file:
        file.write(content)
