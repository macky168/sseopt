def extract_schema(lst):
    lst = [list(x) for x in zip(*lst)]
    schema = [list(set(lst[i])) for i in range(len(lst))]
    return schema
