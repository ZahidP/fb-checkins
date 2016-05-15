

- Consider percentage differences
- Remove all zero spread rows first or later?
- Python Mapping:
    `title_dict = {'male': 'mr.', 'female': 'ms.'}
    table['title'] = map(lambda title,
        gender: title if title != None else title_dict[gender],
        table['title'], table['gender'])`

      `st['a'] = map(lambda path, row: path + 2 * row, st['path'], st['row'])`
