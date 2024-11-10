import gc

import numpy as np
import torch
from probing_utils import load_model_and_validate_gpu, tokenize, generate
from tqdm import tqdm


def compute_correctness_imdb(model_answers, labels):
    name_to_labels = {'negative': 0, 'positive': 1, 'neutral': -1}
    pred = []
    exact_answers = []
    for ans in model_answers:
        if len(ans) == 0:
            pred.append(-1)
            exact_answers.append('NO ANSWER')
            continue
        try:
            if ans.startswith("Label: "):
                ans = ans.split("Label: ")[1]
            exact_answer = ans.split()[0].strip(".</s>")
            pred.append(name_to_labels[exact_answer])
            exact_answers.append(exact_answer)
        except KeyError:
            # Convert the answer to lowercase once
            ans_lower = ans.lower()

            # Find the indices of 'negative' and 'positive'
            negative_index = ans_lower.find('negative')
            positive_index = ans_lower.find('positive')

            # Determine the prediction and exact answer based on the indices
            if negative_index == -1 and positive_index == -1:
                pred.append(-1)
                exact_answers.append('NO ANSWER')
            elif negative_index != -1 and (positive_index == -1 or negative_index < positive_index):
                pred.append(0)
                exact_answers.append('negative')
            else:
                pred.append(1)
                exact_answers.append('positive')
    pred = np.array(pred)
    labels = np.array(labels)

    return {"correctness": (pred == labels).astype(int), "exact_answer": exact_answers}

def compute_correctness_triviaqa(all_textual_answers, labels):
    correctness = []

    for idx in range(len(all_textual_answers)):
        model_answer = all_textual_answers[idx]
        correct = 0
        if type(labels[idx]) == str:
            labels_ = eval(labels[idx])
        else:
            labels_ = labels[idx]
        for ans in labels_:
            if ans.lower() in model_answer.lower():
                correct = 1
                break
        correctness.append(correct)
    return {"correctness": correctness}

def compute_correctness_winobias(model_answers, labels, wrong_labels):
    correctness = []
    exact_answers = []
    for ans, correct_label, incorrect_label in zip(model_answers, labels, wrong_labels):
        ind_ans = ans.lower().find(correct_label.lower())
        ind_inc_ans = ans.lower().find(incorrect_label.lower())
        if (ind_ans == -1) and (ind_inc_ans == -1):
            correctness.append(0)
            print("Problem in answer!")
            print(ans, correct_label, incorrect_label)
            exact_answers.append("")
            continue
        elif (ind_ans != -1) and (ind_inc_ans != -1):
            if ind_ans < ind_inc_ans:
                correctness.append(1)
                exact_answers.append(correct_label)
            else:
                correctness.append(0)
                exact_answers.append(incorrect_label)
            continue
        elif ind_ans != -1:
            correctness.append(1)
            exact_answers.append(correct_label)
            continue
        else:
            correctness.append(0)
            exact_answers.append(incorrect_label)

    return {"correct_labels": labels, "incorrect_answer": wrong_labels, "correctness": correctness, "exact_answer": exact_answers}

def compute_correctness_hotpotqa(model_answers, labels):
    correctness = []
    for model_answer, label in zip(model_answers, labels):
        if label.lower() in model_answer.lower():
            correctness.append(1)
        else:
            correctness.append(0)
    return {"correctness": correctness}


def compute_correctness_math(model_answers, labels):
    correctness = []
    for model_answer, label in zip(model_answers, labels):
        is_correct = (str(label) in model_answer.lower()) or (str(int(label)) in model_answer.lower())
        correctness.append(int(is_correct))

    return {"correctness": correctness}

def compute_correctness_movies(model_answers, labels):
    correctness = []
    for model_answer, label in zip(model_answers, labels):
        if label.lower().strip() in model_answer.lower().strip():
            correctness.append(1)
        else:
            correctness.append(0)
    return {"correctness": correctness}


def compute_correctness_nli(model_answers, labels):
    labels_dict = {
        'neutral': ['neutrality', 'neutral', 'neutality'],
        'entailment': ['entailment', 'entail'],
        'contradiction': ['contradiction', 'contradict']
    }
    correctness = []
    exact_answers = []

    for model_answer, correct_answer in zip(model_answers, labels):
        if correct_answer not in labels_dict:
            print(f"Error: {correct_answer}")
            correctness.append(0)
            continue

        first_label = 'NO_ANSWER'
        min_idx = len(model_answer)
        found_label_str = ""

        for label_name in labels_dict.keys():
            for label_str in labels_dict[label_name]:
                idx = model_answer.lower().find(label_str)
                if idx != -1 and idx < min_idx:
                    first_label = label_name
                    min_idx = idx
                    found_label_str = label_str

        is_correct = int(first_label == correct_answer)
        exact_answers.append(found_label_str)
        if is_correct:
            correctness.append(1)
        else:
            correctness.append(0)

    return {"correctness": correctness,
            'exact_answers': exact_answers}

def compute_correctness_natual_questions(all_questions, model_answers, labels, model=None, tokenizer=None):
    print("Computing correctness for Natural Questions")

    if model is None:
        model, tokenizer = load_model_and_validate_gpu('mistralai/Mistral-7B-Instruct-v0.2')
    correctness = []
    for question, model_answer, label in tqdm(zip(all_questions, model_answers, labels)):
        if str(label).lower() in str(model_answer).lower():
            correctness.append(1)
        else:
            prompt = f"""
                Evaluate the following answers to questions. For each question you would be given a model answer and the correct answer.
                You would have to determine if the model answer is correct or not. If the model answer is correct, write '1' and if it is not correct, write '0'.
                For example:
                
                Question: who is the young guitarist who played with buddy guy?
                Ground Truth: Quinn Sullivan
                Model Answer: Ronnie Earl Explanation: Ronnie Earl is an American blues guitarist and singer who has played with many famous blues musicians, including Buddy Guy. He is known for his soulful and melodic playing style, and has released many albums that blend blues, jazz, and rock music. Earl has also been a member of the Buddy Guy Blues Band and has played with other notable blues musicians such as B.B. King, Eric Clapton, and Stevie Ray Vaughan. He is considered one of the most
                Correctness: 0
                
                Question: name of the first episode of stranger things 
                Ground Truth: Chapter One : The Vanishing of Will Byers
                Model Answer:  The disappearance of Will Byers. Explanation: The first episode of the first season of Stranger Things is titled "The Vanishing of Will Byers". The episode introduces the main characters and sets the tone for the rest of the series. It follows the story of Will Byers, a young boy who goes missing in the fictional town of Hawkins, Indiana, and the subsequent search for him by his mother Joyce and his friends Mike, Dustin, and Lucas. The episode sets the stage for the supernatural
                Correctness: 1
                
                Question: {question}
                Ground Truth: {label}
                Model Answer: {model_answer}
                Correctness:
                """

            model_input = tokenize(prompt, tokenizer, 'mistralai/Mistral-7B-Instruct-v0.2').to(model.device)
            valid = 0
            retries = 0
            sample = True
            while valid == 0 and retries < 5:
                with torch.no_grad():
                    model_output = generate(model_input, model, 'mistralai/Mistral-7B-Instruct-v0.2', sample, False)
                    current_correctness = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])

                current_correctness = (
                    current_correctness.replace(".</s>", "").replace("</s>", "").split('\n')[0].strip().strip("."))
                index_of_1 = current_correctness.find('1')
                index_of_0 = current_correctness.find('0')
                if index_of_1 != -1 and (index_of_0 == -1 or index_of_1 < index_of_0):
                    valid = 1
                    correctness.append(1)
                    break
                elif index_of_0 != -1 and (index_of_1 == -1 or index_of_0 < index_of_1):
                    valid = 1
                    correctness.append(0)
                    break
                else:
                    print(f"Invalid input: {current_correctness}")
                    retries += 1

                sample = True
                retries += 1

            if valid == 0:
                print("Invalid input")
                correctness.append(0)

    return {"correctness": correctness}

def compute_correctness_winogrande(model_answers, labels, wrong_labels, model_name):
    correctness = []
    exact_answers = []

    for model_answer, label, wrong_label in zip(model_answers, labels, wrong_labels):
        if 'llama' in model_name.lower() and 'A)' in model_answer: # The model answer is in the format A) <answer> B) <Answer> ...
            # find the part with the answer
            ans_idx = model_answer.lower().find('answer')
            if ans_idx != -1:
                exact_ans = model_answer[ans_idx+len('answer'):]
                exact_ans = exact_ans.strip()
                if exact_ans.startswith(':'):
                    exact_ans = exact_ans[1:]
                exact_ans = exact_ans.strip()
                if exact_ans.startswith('is'):
                    exact_ans = exact_ans[2:]
                exact_ans = exact_ans.strip()
                exact_ans = exact_ans.split('.')[0]
                model_answer = exact_ans
                print('After cleaning:', exact_ans)


        correct_ans_index = model_answer.lower().find(label.lower())
        wrong_ans_index = model_answer.lower().find(wrong_label.lower())
        if correct_ans_index != -1 and (wrong_ans_index == -1 or correct_ans_index < wrong_ans_index):
            correctness.append(1)
            exact_answers.append(model_answer[correct_ans_index:correct_ans_index + len(label)])
        else:
            correctness.append(0)
            if wrong_ans_index == -1:
                print("Problem in answer!")
                print(model_answer, label, wrong_label)
                exact_answers.append('NO ANSWER')
            else:
                exact_answers.append(model_answer[wrong_ans_index:wrong_ans_index + len(wrong_label)])
    return {"correct_labels": labels, "incorrect_answer": wrong_labels, "correctness": correctness, "exact_answer": exact_answers}

def compute_correctness(all_questions, dataset_name, model_name, labels, model, model_answers, tokenizer, wrong_labels):
    if 'natural_questions' in dataset_name:
        if model == 'mistralai/Mistral-7B-Instruct-v0.2':
            res = compute_correctness_natual_questions(all_questions, model_answers, labels, model=model,
                                                               tokenizer=tokenizer)
        else:
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            res = compute_correctness_natual_questions(all_questions, model_answers, labels)
    elif 'winogrande' in dataset_name:
        res = compute_correctness_winogrande(model_answers, labels, wrong_labels, model_name=model_name)
    elif 'winobias' in dataset_name:
        res = compute_correctness_winobias(model_answers, labels, wrong_labels)
    else:
        res = CORRECTNESS_FN[dataset_name.replace("_test", "")](model_answers, labels)
    return res

CORRECTNESS_FN = {
    'triviaqa': compute_correctness_triviaqa,
    'imdb': compute_correctness_imdb,
    'winobias': compute_correctness_winobias,
    'winogrande': compute_correctness_winogrande,
    'hotpotqa': compute_correctness_hotpotqa,
    'hotpotqa_with_context': compute_correctness_hotpotqa,
    'math': compute_correctness_math,
    'movies': compute_correctness_movies,
    'mnli': compute_correctness_nli,
    'natural_questions_with_context': compute_correctness_natual_questions
}