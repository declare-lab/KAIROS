# Utility functions to compute Exact Match and Citation Scores to reward
import csv
import os
import re
import traceback
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PATTERN = r"^<think>\n(.*?)\n</think>\n\n<answer>\n(.*?)\n</answer>\s*\Z"

def format_reward(text: str):
    match = re.match(PATTERN, text, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0


def count_tags(text: str) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.25
    if text.count("\n</think>\n") == 1:
        count += 0.25
    if text.count("\n<answer>\n") == 1:
        count += 0.25
    if text.count("\n</answer>") == 1:
        count += 0.25
    return count


def split_reasoning(text: str):
    match = re.search(PATTERN, text, re.DOTALL)
    if not match:
        return None, None
    think_text, answer_text = match.groups()
    return think_text.strip(), answer_text.strip()


def get_voices(text:str):
    # strict matching for voice may not be good since
    matches = re.findall(r'^(\w+)\s+voice:', text.lower(), re.MULTILINE)
    return matches

def group_similar_items(items, sim_matrix, threshold=0.75):
    n = len(items)
    visited = set()
    groups = []

    for i in range(n):
        if i in visited:
            continue
        group = [items[i]]
        visited.add(i)
        for j in range(i + 1, n):
            if j not in visited and sim_matrix[i][j] >= threshold:
                group.append(items[j])
                visited.add(j)
        groups.append(tuple(group))
    return groups


model = SentenceTransformer('all-MiniLM-L6-v2') 

def get_unique_voices(inner_voices: List[str]):
    # Get Unique Voices
    unique_inner_voices = list(set(inner_voices))

    if len(unique_inner_voices)==1:
        return [(unique_inner_voices[0])]

    # Calculate Similarity
    embeddings = model.encode(unique_inner_voices)
    similarity_matrix = cosine_similarity(embeddings)
    
    # CLuster Voices Based on Similarity 
    voice_groups = group_similar_items(unique_inner_voices, similarity_matrix)

    return voice_groups

# import spacy

# # Load spaCy model with word vectors
# nlp = spacy.load("en_core_web_md")

# def group_similar_voices_spacy(voices, threshold=0.75):
#     docs = [nlp(voice) for voice in voices]
#     n = len(voices)
#     visited = set()
#     groups = []

#     for i in range(n):
#         if i in visited:
#             continue
#         group = [voices[i]]
#         visited.add(i)
#         for j in range(i + 1, n):
#             if j not in visited and docs[i].similarity(docs[j]) >= threshold:
#                 group.append(voices[j])
#                 visited.add(j)
#         groups.append(tuple(group))
#     return groups

def has_non_consecutive_turn(groups, sequence):
    # Build map from item to group index
    item_to_group = {}
    for i, group in enumerate(groups):
        for item in group:
            item_to_group[item] = i

    # Get group sequence from item sequence
    group_sequence = [item_to_group[item] for item in sequence if item in item_to_group]

    seen = {}
    for i, g in enumerate(group_sequence):
        if g not in seen:
            seen[g] = i
        else:
            # Check if any different group appeared in between
            if any(group_sequence[j] != g for j in range(seen[g] + 1, i)):
                return True
    return False


def get_boxed(text):
    """
    Extracts the text inside \boxed{} from a given string.

    Args:
    text: The input string potentially containing \boxed{}

    Returns:
    The text inside the first \boxed{} found, or None if not found.
    """
    match = re.search(r'\\boxed{(.*?)}', text)
    if match:
        return match.group(1).strip()
    return None

def is_conversational(messages):
    if isinstance(messages, list):
        message = messages[0]
        # Each message must a list of dictionaries with keys "role" and "content"
        if isinstance(message, dict) and "role" in message and "content" in message:
            return True
    return False

def get_option_char(model_answer):
    try:
        tmp=model_answer.split('is: "(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: (')
        if len(tmp) == 1:
            tmp = model_answer.split('is (')
        assert len(tmp) > 1, "model didn't output trigger"
        assert tmp[-1][1] == ')', "didnt output letter for choice"
        pred = tmp[-1][0]
        return pred
    except Exception as e:
        return traceback.format_exc()


def combined_reward(completions, **kwargs):
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]

    rewards = []
    answers = [gt_option.strip() for gt_option in kwargs['gt_option']]

    for completion, answer in zip(completions, answers):
        reward = 0

        # Strict Format Reward
        format_score = format_reward(completion)

        if format_score == 1:
            reward += format_score # --> +1

            reasoning, pred_answer = split_reasoning(completion)

            # correctness reward --> extract from \boxed{}
            pred_answer_option = get_option_char(pred_answer) 
            if pred_answer_option:
                if answer[1] == pred_answer_option:
                    reward += 1 # +1 correctness
            
            # Inner Voice Reward
            inner_voices = get_voices(reasoning)

            if len(inner_voices) > 0: # Total Inner Voice reward range 0-1
                reward += 0.25 #reward for having inner voice

                voice_groups = get_unique_voices(inner_voices)

                # At least 2 distinct roles of voices
                if len(voice_groups) > 1:
                    reward += 0.25

                    voice_to_group = {}
                    for i, group in enumerate(voice_groups):
                        for voice in group:
                            voice_to_group[voice] = i

                    # Get group sequence from voice sequence
                    group_sequence = [voice_to_group[voice] for voice in inner_voices if voice in voice_to_group]

                    is_consequetive = any([group_sequence[idx]==group_sequence[idx+1] for idx in range(len(group_sequence)-1)])

                    # Non Consequetive Turns
                    if not is_consequetive:
                        reward += 0.25
                        
                        # atleast 1 voice group has multiple turn
                        if len(set(group_sequence)) < len(group_sequence):
                            reward += 0.25
        else:
            # tag reward
            reward += count_tags(completion) # --> 0 - 1

        rewards.append(reward)
        # print(completion)
        # print(reward)

    if len(rewards) != len(completions):
        breakpoint()

    return rewards


def format_reward_normal(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    matches = [re.match(PATTERN, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def accuracy_reward_normal(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    rewards = []
    answers = [gt_option.strip() for gt_option in kwargs['gt_option']]
    
    for content, answer in zip(completions, answers):
        
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        answer_parsed = answer_match.group(1).strip() if answer_match else ""
        pred_answer_option = get_option_char(answer_parsed) 
        
        if answer[1] == pred_answer_option:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards
        

if __name__ == "__main__":
    output = """<think>

</think>

<answer>
I'm leaning towards "C) I have no comment" as the best answer. The other options seem too specific and biased, and "I have no comment" is a more neutral and appropriate response to a question about the government lying to us. It's a fact that the government often doesn't want to admit to certain things or provide accurate information, but it's not something that can be proven or verified. The other options are either too negative or too specific, and "I have no comment" seems like the most fitting choice.
</answer>"""

    print(count_tags(output))
    print(format_reward(output))
    
    output = """<think>\nxxxxxxx\n</think>\n\n<answer>\nYou: The best answer is: "(B) the content of the answer"\n</answer>"""
    
    print(format_reward_normal([output]))
    print(accuracy_reward_normal([output], gt_option=["(A) 730 cm cube"]))
    # pass
    # Test Extraction
    # Completion = "<think>\n Reasoning \n Trace \n</think>\n\n<answer>\nAnswer\n</answer>"
    # print(Completion)
    # print(format_reward(Completion))
    # print(count_tags(Completion))
    # print(split_reasoning(Completion))

#     Reasoning = """
# iauhbfio augbioah 

# hungry Voice: sidfhos

# Indulgent Voice: ...

# Disciplined Voice: ...

# Hungry Voice: ...
# """
#     voices = get_voices(Reasoning) 
#     print(voices)

#     uniq = get_unique_voices(voices)
#     uniq = group_similar_voices_spacy(voices)
#     print(uniq)

#     print(has_non_consecutive_turn(uniq, voices))


    # # Test Get Unique Voices
    # print(group_similar_voices_spacy([
    # "disciplined",
    # "self-control",
    # "gluttonous",
    # "lazy",
    # "responsible",
    # "indulgent",
    # "productive",
    # "hungry",
    # "hard working"
    # ]))

    # # Test turn checking fn
    # groups = [('A', 'B'), ('C', 'D', 'E')]
    # sequence = ['A', 'B', 'C', 'D', 'E', 'A', 'D', 'E']
    # print(has_non_consecutive_turn(groups, sequence), "Should be True")

    # groups = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    # sequence = ['A', 'C', 'E', 'A'] 
    # print(has_non_consecutive_turn(groups, sequence), "Should be True") 

    # groups = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    # sequence = ['A', 'C', 'A', 'E'] 
    # print(has_non_consecutive_turn(groups, sequence), "Should be True") 

    # groups = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    # sequence = ['A', 'B', 'C', 'D', 'E', 'F']
    # print(has_non_consecutive_turn(groups, sequence), "Should be False") 

