# import t2v_metrics
# from typing import List, Dict
# import json
# import torch

# class FineVQAReward():
#     def __init__(self, model='clip-flant5-xxl', device='cuda', log_file='./log.jsonl'):
#         self.model = t2v_metrics.VQAScore(model=model, cache_dir="./reward_model/clip-flant5-xxl")
#         self.device = device
#         self.reward_model = self.model.to(self.device)
#         self.reward_model.eval()

#         self.log_file = log_file
#         with open(self.log_file, 'w', encoding='utf-8') as f:
#             pass

#     def __call__(self, images: List[str], scenes: List[Dict]):
#         vqa_score_list = []
#         for image, scene in zip(images, scenes):
#             questions = []
#             dependencies = []
#             question_types = []

#             # Process each category
#             categories = ['object', 'count', 'attribute', 'relation']
#             for category in categories:
#                 for item in scene['qa'][category]:
#                     questions.append(item['question'])
#                     dependencies.append(item['dependencies'])
#                     question_types.append(category)

#             support_data = {
#                 'questions': questions,
#                 'dependencies': dependencies,
#                 'question_types': question_types
#             }
            
#             vqa_score = self.reward_model([image], support_data['questions'])[0]
#             vqa_score = vqa_score.tolist()
#             sum_score = 0
#             for score, dependency, question_type in zip(vqa_score, dependencies, question_types):
#                 if question_type == 'object':
#                     sum_score += score
#                 elif question_type == 'attribute' or question_type == 'count':
#                     try:
#                         sum_score += score * vqa_score[dependency[0] - 1]
#                     except IndexError as e:
#                         print(f"vqascore:{vqa_score}, type:{question_types}, dependency:{dependency}")
#                 elif question_type == "relation":
#                     try:
#                         sum_score += score * (min(vqa_score[dependency[0] - 1], vqa_score[dependency[1] - 1]))
#                     except IndexError as e:
#                         print(f"vqascore:{vqa_score}, type:{question_types}, dependency:{dependency}")
#                 else:
#                     raise ValueError("Not implemented question type error")
            
#             avg_vqa_score = sum_score / len(questions) # assume questions is not an empty list
#             vqa_score_list.append(avg_vqa_score)
#             log_data = {
#                 "image_path": image,
#                 "difficulty": scene['difficulty'],
#                 "prompt": scene['prompt'],
#                 "questions": questions,
#                 "vqa_score": vqa_score,
#                 "avg_vqa_score": avg_vqa_score
#             }

#             with open(self.log_file, 'a', encoding='utf-8') as f:
#                 f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
            
#         return torch.tensor(vqa_score_list, dtype=torch.float32, device=self.device)

import json
import os
from typing import Dict, List
import t2v_metrics
import torch
from t2v_metrics.models.vqascore_models import clip_t5_model, llava16_model


class FineVQAReward:
    def __init__(self, model="clip-flant5-xxl", device="cuda", log_file="./log_testcurr.jsonl", 
                 use_fine_grained=True, use_calibration=True):
        
        assert not use_calibration or use_fine_grained, "use_calibration can only be True when use_fine_grained=True"
        self.use_fine_grained = use_fine_grained
        self.use_calibration = use_calibration
        
        # Set question template based on fine_grained setting
        if self.use_fine_grained:
            # clip_t5_model.default_question_template = "{} Please answer yes or no."
            llava16_model.default_question_template = '{} Answer only "yes" or "no", one word only.'
        else:
            # clip_t5_model.default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
            llava16_model.default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
        
        if model == "llava-v1.6-13b":
            cache_dir = "./reward_model/llava-v1.6"
        elif model == "clip-flant5-xxl":
            cache_dir = "./reward_model/clip-flant5-xxl"
        elif model == "llava-v1.5-13b":
            cache_dir = "./reward_model/llava-v1.5"
        elif model == "instructblip-flant5-xxl":
            cache_dir = "./reward_model/instructblip-flan-t5-xxl"
        elif model == "llava-v1.5-7b":
            cache_dir = "./reward_model/llava-v1.5-7b"
        else:
            cache_dir = ""
            raise ValueError("Not implemented reward model")

        self.model = t2v_metrics.VQAScore(model=model, cache_dir=cache_dir)
        self.device = device
        self.reward_model = self.model.to(self.device)
        self.reward_model.eval()

        self.log_file = log_file
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            pass

    def __call__(self, images: List[str], scenes: List[Dict]):
        vqa_score_list = []
        for image, scene in zip(images, scenes):
            if self.use_fine_grained:
                # Use fine-grained questions from qa
                questions = []
                dependencies = []
                question_types = []

                # Process each category (skip if not present in qa)
                categories = ["object", "count", "attribute", "relation"]
                for category in categories:
                    for item in scene["qa"].get(category, []):
                        questions.append(item["question"])
                        dependencies.append(item.get("dependencies", []))
                        question_types.append(category)

                support_data = {"questions": questions, "dependencies": dependencies, "question_types": question_types}
                raw_vqa_score = self.reward_model([image], support_data["questions"])[0].tolist()
                
                if self.use_calibration:
                    # Apply calibration and record calibrated scores
                    vqa_score = []
                    sum_score = 0
                    for score, dependency, question_type in zip(raw_vqa_score, dependencies, question_types):
                        if question_type == "object":
                            calibrated_score = score
                        elif question_type == "attribute" or question_type == "count":
                            try:
                                calibrated_score = score * raw_vqa_score[dependency[0] - 1]
                            except IndexError as e:
                                print(f"vqascore:{raw_vqa_score}, type:{question_types}, dependency:{dependency}")
                                calibrated_score = score
                        elif question_type == "relation":
                            try:
                                calibrated_score = score * (min(raw_vqa_score[dependency[0] - 1], raw_vqa_score[dependency[1] - 1]))
                            except IndexError as e:
                                print(f"vqascore:{raw_vqa_score}, type:{question_types}, dependency:{dependency}")
                                calibrated_score = score
                        else:
                            raise ValueError("Not implemented question type error")
                        
                        vqa_score.append(calibrated_score)
                        sum_score += calibrated_score
                    
                    avg_vqa_score = sum_score / len(questions)
                else:
                    # No calibration, use raw scores directly
                    vqa_score = raw_vqa_score
                    avg_vqa_score = sum(vqa_score) / len(questions)
                
            else:
                # Use prompt
                questions = [scene["prompt"]]
                vqa_score = self.reward_model([image], questions).item()
                avg_vqa_score = vqa_score

            vqa_score_list.append(avg_vqa_score)
            
            log_data = {
                "image_path": image,
                "difficulty": scene["difficulty"],
                "prompt": scene["prompt"],
                "questions": questions,
                "vqa_score": vqa_score,
                "avg_vqa_score": avg_vqa_score,
                "use_fine_grained": self.use_fine_grained,
                "use_calibration": self.use_calibration,
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        return torch.tensor(vqa_score_list, dtype=torch.float32, device=self.device)

