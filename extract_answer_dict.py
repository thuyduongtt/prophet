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


def extract_answer_dict(path_to_dataset, output_file_name, split='train'):
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

    selected_answers = [ans for ans in count_answer.keys() if count_answer[ans] >= 10]
    removed_answers = [ans for ans in count_answer.keys() if count_answer[ans] < 10]

    print('Total unique answers:', len(np.unique(all_answers)))
    print('Total selected answers:', len(selected_answers), '| removed answers:',  len(removed_answers))

    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump(selected_answers, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_file_name', type=str, required=True, help='Path to dataset')
    args = parser.parse_args()

    extract_answer_dict(args.path_to_ds, args.output_file_name)
