import nltk

def convert_dataset():
    data_file_path = "./data/conala-train.json"

    data = preprocess_conala_dataset(data_file_path)

def preprocess_conala_dataset(data_path):
    examples = []
    with open('./data-parsed/conala_dataset.examples.txt', 'w') as output_examples:
        with open(data_path, 'r') as date:
            for idx, data_example in enumerate(date):
                clean_query_tokens, clean_code, parse_tree = canonicalize_conala_example(data_example)
                # example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code, 'parse_tree': parse_tree,
                #            'str_map': None, 'raw_code': code}
                # examples.append(example)
                #
                # output_examples.write('*' * 50 + '\n')
                # output_examples.write('example# %d\n' % idx)
                # output_examples.write(' '.join(clean_query_tokens) + '\n')
                # output_examples.write('\n')
                # output_examples.write(clean_code + '\n')
                # output_examples.write('*' * 50 + '\n')
                #
                # idx += 1


    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples

def canonicalize_conala_example(data_example):
    data_example = re.sub(r'<.*?>', '', data_example)
    data_example_tokens = nltk.word_tokenize(data_example)

    # code = code.replace('§', '\n').strip()
    #
    # # sanity check
    # parse_tree = parse_raw(code)
    # gold_ast_tree = ast.parse(code).body[0]
    # gold_source = astor.to_source(gold_ast_tree)
    # ast_tree = parse_tree_to_python_ast(parse_tree)
    # pred_source = astor.to_source(ast_tree)
    #
    # assert gold_source == pred_source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, pred_source)
    #
    # return query_tokens, code, parse_tree

def main():
    convert_dataset()

if __name__ == '__main__':
    main()
