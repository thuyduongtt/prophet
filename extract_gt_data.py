import argparse
import json
import ijson
import pickle
from pathlib import Path
import numpy as np

'''
n_questions: int
exported_time: datetime
questions: array
    image_id
    image_name
    image_dir
    dataset_name
    question_id
    question
    answers
    answers_scores
    choices
    choice_scores
    property_id
    property_label
    n_hop
    has_scene_graph
'''


def stream_data(path_to_json_file, limit=0, start_at=0):
    i = 0
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


# from this we get unbalanced_annotations.json and balanced_10_annotations.json
def extract_questions(path_to_dataset, ds_name, split='train'):
    json_data = stream_data(f'{path_to_dataset}/{split}.json')
    questions = []

    for d in json_data:
        questions.append({
            'image_id': d['image_id'],
            'question_id': d['question_id'],
            'question': d['question']
        })

    with open(f'assets/{ds_name}_{split}_questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f)


# from this we get unbalanced_annotations.json and balanced_10_annotations.json
def extract_answer_annotation(path_to_dataset, ds_name, split='train'):
    json_data = stream_data(f'{path_to_dataset}/{split}.json')
    answers = []

    for d in json_data:
        answers.append({
            'image_id': d['image_id'],
            'question_id': d['question_id'],
            'answers': [{
                'answer_id': i + 1,
                'answer': d['answers'][i]
            } for i in range(len(d['answers']))]
        })

    with open(f'assets/{ds_name}_{split}_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(answers, f)


# from this we get answer_dict_unbalanced.json and answer_dict_balanced_10.json
def extract_answer_dict(path_to_dataset, ds_name, split='train'):
    json_data = stream_data(f'{path_to_dataset}/{split}.json')
    all_answers = []

    for d in json_data:
        all_answers += d['answers']

    print('Total answers:', len(all_answers))

    # remove answers that appear less than 10 times
    count_answer = {}

    for ans in all_answers:
        if ans not in count_answer:
            count_answer[ans] = 0
        count_answer[ans] += 1

    # make sure the final number of answer is larger than 3129 !!!
    BOUND = 3  # see section 3.5 in this paper: https://arxiv.org/pdf/1708.02711.pdf

    selected_answers = [ans for ans in count_answer.keys() if count_answer[ans] >= BOUND]
    removed_answers = [ans for ans in count_answer.keys() if count_answer[ans] < BOUND]

    print('Total unique answers:', len(np.unique(all_answers)))
    print('Total selected answers:', len(selected_answers), '| removed answers:', len(removed_answers))

    with open(f'assets/answer_dict_{ds_name}.json', 'w', encoding='utf-8') as f:
        json.dump(selected_answers, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    # extract_questions(args.path_to_ds, args.ds_name, args.split)
    # extract_answer_annotation(args.path_to_ds, args.ds_name, args.split)
    extract_answer_dict(args.path_to_ds, args.ds_name, args.split)
