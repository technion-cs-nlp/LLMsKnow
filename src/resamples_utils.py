from collections import defaultdict
import numpy as np

def get_error_stats(textual_answers, exact_answers, model_output_greedy, compute_correctness_fn):
    textual_answers_per_question = []
    for i in range(len(textual_answers[0])):
        textual_answers_per_question.append([textual_answers[j][i] for j in range(len(textual_answers))])
    results = []
    for q_idx, greedy_correctness in zip(range(len(textual_answers_per_question)),
                                         model_output_greedy['automatic_correctness']):

        answers = textual_answers_per_question[q_idx]
        question = model_output_greedy.iloc[q_idx]['question']

        correct_cluster = []
        others_idx = []
        others_exact_answers = []
        others = []
        for retry_idx, retry_ans in enumerate(answers):
            if 'incorrect_answer' in model_output_greedy:
                correct = compute_correctness_fn([retry_ans],
                                                 [[model_output_greedy.iloc[q_idx].correct_answer],
                                                  [model_output_greedy.iloc[q_idx].incorrect_answer]])['correctness'][0]
            else:
                correct = compute_correctness_fn([retry_ans], [model_output_greedy.iloc[q_idx].correct_answer])['correctness'][0]
            if correct:
                correct_cluster.append(retry_ans)
            else:
                others_idx.append(retry_idx)
                others.append(retry_ans)

        for retry_idx in others_idx:
            if not exact_answers['valid_exact_answer'][q_idx][retry_idx]:
                continue
            exact_answer = exact_answers['exact_answer'][q_idx][retry_idx]
            others_exact_answers.append(exact_answer)

        ctr_other_exact_answers = defaultdict(int)
        for ans in others_exact_answers:
            flag = 0
            if ans in ctr_other_exact_answers:
                ctr_other_exact_answers[ans] += 1
                flag = 1
            else:
                for existing_answer in ctr_other_exact_answers:
                    if (ans.lower() in existing_answer.lower()) or (existing_answer.lower() in ans.lower()):
                        ctr_other_exact_answers[existing_answer] += 1
                        flag = 1
            if flag == 0:
                ctr_other_exact_answers[ans] += 1

        results.append(
            {
                "question": question,
                "greedy_correctness": greedy_correctness,
                "n_wrong_answers": len(ctr_other_exact_answers.keys()),
                "correct_answer_size": len(correct_cluster),
                "largest_incorrect_answer_size": np.max(list(ctr_other_exact_answers.values())) if len(
                    ctr_other_exact_answers) > 0 else 0,
                "wrong_answers": dict(ctr_other_exact_answers),
                "wrong_answers_raw": others,
                "correct_answers_raw": correct_cluster,
                "correct_answer": model_output_greedy.iloc[q_idx].correct_answer
            }
        )

    return results

def get_types_of_mistakes(results, n_resamples):
    is_largest_no_answer = get_is_largest_no_answer(results)

    half_of_samples = n_resamples // 2
    third_of_samples = n_resamples // 3
    sixth_of_samples = n_resamples // 6

    TYPES_OF_MISTAKES = {
        "no_answer_is_largest": is_largest_no_answer,
        "wrong_is_largest_1": ((results.largest_incorrect_answer_size >= half_of_samples) & (is_largest_no_answer == 0) & (
                results.correct_answer_size > 0)).to_numpy(),
        "wrong_is_largest_2": ((results.largest_incorrect_answer_size >= half_of_samples) & (is_largest_no_answer == 0) & (
                results.correct_answer_size == 0)).to_numpy(),
        "right_is_largest_1": ((results.correct_answer_size >= half_of_samples) & (results.n_wrong_answers > 0)).to_numpy(),
        "right_is_largest_2": (results.correct_answer_size == n_resamples).to_numpy(),
        "many_different_answers_1": ((results.n_wrong_answers >= (third_of_samples - 1)) & (results.correct_answer_size > 0)).to_numpy(),
        "many_different_answers_2": ((results.n_wrong_answers >= third_of_samples) & (results.correct_answer_size == 0)).to_numpy(),
        "closely_competing_answers": (
                ((results['correct_answer_size'] - results['largest_incorrect_answer_size']).abs() <= sixth_of_samples) & (
                results['correct_answer_size'] > sixth_of_samples) & (
                        results['largest_incorrect_answer_size'] > sixth_of_samples)).to_numpy(),
        "all": np.array([True] * len(results))
    }
    return TYPES_OF_MISTAKES
def get_is_largest_no_answer(results):
    is_largest_no_answer = []
    for wrong_answers, correct_size in zip(results['wrong_answers'], results['correct_answer_size']):
        wrong_answers_ = eval(wrong_answers) if type(wrong_answers) == str else wrong_answers
        if ('NO ANSWER' in wrong_answers_) and (wrong_answers_['NO ANSWER'] == max(wrong_answers_.values()) and (
                wrong_answers_['NO ANSWER'] > correct_size)):
            is_largest_no_answer.append(True)
        else:
            is_largest_no_answer.append(False)
    is_largest_no_answer = np.array(is_largest_no_answer)
    return is_largest_no_answer