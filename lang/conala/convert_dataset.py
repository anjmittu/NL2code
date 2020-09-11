import nltk
import re
import json
from itertools import chain
import logging
import string
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from lang.py.parse import parse, parse_tree_to_python_ast, canonicalize_code, get_grammar, parse_raw, \
    de_canonicalize_code, tokenize_code, tokenize_code_adv, de_canonicalize_code_for_seq2seq
from lang.py.unaryclosure import get_top_unary_closures, apply_unary_closures
from dataset import gen_vocab, DataSet, DataEntry, Action, APPLY_RULE, GEN_TOKEN, COPY_TOKEN, GEN_COPY_TOKEN, Vocab
import ast
import astor
from nn.utils.io_utils import serialize_to_file, deserialize_from_file

def convert_dataset():
    MAX_QUERY_LENGTH = 70
    WORD_FREQ_CUT_OFF = 3
    data_file_path = os.path.join(PROJECT_ROOT, "lang/conala/data")

    # pre-process conala training data
    data = preprocess_conala_dataset(os.path.join(data_file_path, "conala-train.json"))
    # pre-process conala test data
    data.extend(preprocess_conala_dataset(os.path.join(data_file_path, "conala-test.json")))
    parse_trees = [e['parse_tree'] for e in data]

    # apply unary closures
    unary_closures = get_top_unary_closures(parse_trees, k=20)
    for parse_tree in parse_trees:
        apply_unary_closures(parse_tree, unary_closures)

    # build the grammar
    grammar = get_grammar(parse_trees)

    # with open('conala.grammar.unary_closure.txt', 'w') as f:
    #     for rule in grammar:
    #         f.write(rule.__repr__() + '\n')

    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=WORD_FREQ_CUT_OFF)

    print('annot vocab. size: %d', annot_vocab.size)

    def get_terminal_tokens(_terminal_str):
        """
        get terminal tokens
        break words like MinionCards into [Minion, Cards]
        """
        tmp_terminal_tokens = [t for t in _terminal_str.split(' ') if len(t) > 0]
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            sub_tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).split(' ')
            _terminal_tokens.extend(sub_tokens)

            _terminal_tokens.append(' ')

        return _terminal_tokens[:-1]

    # enumerate all terminal tokens to build up the terminal tokens vocabulary
    all_terminal_tokens = []
    for entry in data:
        parse_tree = entry['parse_tree']
        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    assert len(terminal_token) > 0
                    all_terminal_tokens.append(terminal_token)

    terminal_vocab = gen_vocab(all_terminal_tokens, vocab_size=5000, freq_cutoff=WORD_FREQ_CUT_OFF)

    # now generate the dataset
    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'conala.train_data')
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'conala.dev_data')
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'conala.test_data')

    all_examples = []

    can_fully_reconstructed_examples_num = 0
    examples_with_empty_actions_num = 0

    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        parse_tree = entry['parse_tree']

        rule_list, rule_parents = parse_tree.get_productions(include_value_node=True)

        actions = []
        can_fully_reconstructed = True
        rule_pos_map = dict()

        for rule_count, rule in enumerate(rule_list):
            if not grammar.is_value_node(rule.parent):
                assert rule.value is None
                parent_rule = rule_parents[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0

                rule_pos_map[rule] = len(actions)

                d = {'rule': rule, 'parent_t': parent_t, 'parent_rule': parent_rule}
                action = Action(APPLY_RULE, d)

                actions.append(action)
            else:
                assert rule.is_leaf

                parent_rule = rule_parents[(rule_count, rule)][0]
                parent_t = rule_pos_map[parent_rule]

                terminal_val = rule.value
                terminal_str = str(terminal_val)
                terminal_tokens = get_terminal_tokens(terminal_str)

                # assert len(terminal_tokens) > 0

                for terminal_token in terminal_tokens:
                    term_tok_id = terminal_vocab[terminal_token]
                    tok_src_idx = -1
                    try:
                        tok_src_idx = query_tokens.index(terminal_token)
                    except ValueError:
                        pass

                    d = {'literal': terminal_token, 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}

                    # cannot copy, only generation
                    # could be unk!
                    if tok_src_idx < 0 or tok_src_idx >= MAX_QUERY_LENGTH:
                        action = Action(GEN_TOKEN, d)
                        if terminal_token not in terminal_vocab:
                            if terminal_token not in query_tokens:
                                # print terminal_token
                                can_fully_reconstructed = False
                    else:  # copy
                        if term_tok_id != terminal_vocab.unk:
                            d['source_idx'] = tok_src_idx
                            action = Action(GEN_COPY_TOKEN, d)
                        else:
                            d['source_idx'] = tok_src_idx
                            action = Action(COPY_TOKEN, d)

                    actions.append(action)

                d = {'literal': '<eos>', 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}
                actions.append(Action(GEN_TOKEN, d))

        if len(actions) == 0:
            examples_with_empty_actions_num += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, code, actions, {'str_map': None, 'raw_code': entry['raw_code']})

        if can_fully_reconstructed:
            can_fully_reconstructed_examples_num += 1

        # train, valid, test splits
        if 0 <= idx < 533:
            train_data.add(example)
        elif idx < 599:
            dev_data.add(example)
        else:
            test_data.add(example)

        all_examples.append(example)

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    # serialize_to_file([len(e.query) for e in all_examples], 'query.len')
    # serialize_to_file([len(e.actions) for e in all_examples], 'actions.len')

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_reconstructed_examples_num, len(all_examples),
                 can_fully_reconstructed_examples_num / len(all_examples))
    logging.info('empty_actions_count: %d', examples_with_empty_actions_num)

    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    train_data.init_data_matrices(max_query_length=70, max_example_action_num=350)
    dev_data.init_data_matrices(max_query_length=70, max_example_action_num=350)
    test_data.init_data_matrices(max_query_length=70, max_example_action_num=350)

    serialize_to_file((train_data, dev_data, test_data),
                      os.path.join(PROJECT_ROOT, 'data/conala.freq{WORD_FREQ_CUT_OFF}.max_action350.pre_suf.unary_closure.bin'.format(WORD_FREQ_CUT_OFF=WORD_FREQ_CUT_OFF)))

    return train_data, dev_data, test_data

def preprocess_conala_dataset(data_path):
    examples = []
    with open(data_path, 'r') as data:
        data_json = json.load(data)
        for idx, data_example in enumerate(data_json):
            try:
                clean_query_tokens, clean_code, parse_tree = canonicalize_conala_example(data_example)
            except Exception:
                print("Unable to parse example")
                print(data_example)
                continue
            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code, 'parse_tree': parse_tree,
                       'str_map': None, 'raw_code': data_example["snippet"]}
            examples.append(example)
            idx += 1


    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples

def canonicalize_conala_example(data_example):
    query = re.sub(r'<.*?>', '', data_example["rewritten_intent"] if data_example["rewritten_intent"] else data_example["intent"])
    query_tokens = nltk.word_tokenize(query)

    # sanity check
    parse_tree = parse_raw(data_example["snippet"])
    gold_ast_tree = ast.parse(data_example["snippet"]).body[0]
    gold_source = astor.to_source(gold_ast_tree)
    ast_tree = parse_tree_to_python_ast(parse_tree)
    pred_source = astor.to_source(ast_tree)

    assert gold_source == pred_source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, pred_source)

    return query_tokens, data_example["snippet"], parse_tree

def main():
    convert_dataset()

if __name__ == '__main__':
    main()
