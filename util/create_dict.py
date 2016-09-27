# -*- coding: utf-8 -*-
import os
import codecs

data_path = '../data'
word_items = list()

for class_name in ['celebrity']:
    entity_file = codecs.open(os.path.join(data_path, class_name + '.ENTITYSET.txt'), 'rb', 'gb18030')
    for line in entity_file.readlines():
        para_index = line.find('(')
        word_items.append(line[:para_index] + '\n')
    entity_file.close()

utf_file = codecs.open(os.path.join(data_path, class_name + '.ENTITYSET-UTF8.txt'), 'wb', 'utf-8')

for i in xrange(1900, 2018):
    word_items.append(str(i)+'\n')

utf_file.writelines(word_items)
utf_file.close()
