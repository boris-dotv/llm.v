"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu
"""

import os
from datasets import load_from_disk
from tasks.common import Task, render_mc

# 使用本地数据路径
MMLU_BASE_DIR = "/public_hw/share/cit_ztyu/cz/nanochat/base_data"

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"

        # 使用本地数据
        if subset == "auxiliary_train":
            ds_path = os.path.join(MMLU_BASE_DIR, "mmlu_auxiliary_train")
            self.ds = load_from_disk(ds_path)
        else:
            ds_path = os.path.join(MMLU_BASE_DIR, "mmlu_all_test")
            self.ds = load_from_disk(ds_path)

        self.subset = subset
        self.split = split

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]

        # auxiliary_train 格式不同：数据在 'train' 列里
        if self.subset == "auxiliary_train":
            data = row['train']
            question = data.get('question', '')
            choices = data.get('choices', [])
            answer = data.get('answer', 0)
            subject = data.get('subject', '')
        else:
            # all test 格式：直接在 row 中
            question = row.get('question', '')
            choices = row.get('choices', [])
            answer = row.get('answer', 0)
            subject = row.get('subject', '')

        # 确保 choices 是列表
        if not isinstance(choices, list):
            choices = list(choices)

        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer] if isinstance(answer, int) else str(answer)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message


# """
# The MMLU dataset.
# https://huggingface.co/datasets/cais/mmlu
# """

# from datasets import load_dataset
# from tasks.common import Task, render_mc

# class MMLU(Task):

#     letters = ('A', 'B', 'C', 'D')
#     groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

#     def __init__(self, subset, split, **kwargs):
#         super().__init__(**kwargs)
#         assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
#         assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
#         if subset == "auxiliary_train":
#             assert split == "train", "auxiliary_train must be split into train"
#         self.subset = subset
#         self.split = split
#         self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
#         if subset == "auxiliary_train":
#             # I don't understand why but the auxiliary_train rows have some weird additional 'train' wrapper
#             self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])

#     @property
#     def eval_type(self):
#         return 'categorical'

#     def num_examples(self):
#         return len(self.ds)

#     def get_example(self, index):
#         row = self.ds[index]
#         question = row["question"] # the question text
#         choices = row["choices"] # the text of each choice
#         answer = row["answer"] # index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
#         subject = row["subject"] # e.g. "college_biology", "college_chemistry", etc.
#         assert len(choices) == 4, "MMLU should have 4 choices"
#         # create and return the Conversation object
#         user_message = render_mc(question, self.letters, choices)
#         assistant_message = self.letters[answer]
#         messages = [
#             {"role": "user", "content": user_message},
#             {"role": "assistant", "content": assistant_message}
#         ]
#         conversation = {
#             "messages": messages,
#             "subject": subject, # might be useful later for grouping metrics by subject
#             "letters": self.letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
#         }
#         return conversation

#     def evaluate(self, conversation, assistant_response):
#         # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
#         # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
#         assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
#         assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
#         return assistant_response == assistant_message
