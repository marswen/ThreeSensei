import os
import json
import utils
import datetime
from llms import load_llm
from langchain import LLMChain, PromptTemplate

task_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
logger = utils.init_logger(task_id)


class ThreeSensei:

    def __init__(self, lang):
        self.lang = lang
        if lang == 'zh':
            from prompts_zh import (
                choose_role_prompt,
                create_identity_prompt,
                make_comment_prompt,
                summarize_analysis_prompt,
                supplement_info_prompt,
                refine_summary_prompt,
                summarize_record_prompt
            )
        else:
            from prompts_en import (
                choose_role_prompt,
                create_identity_prompt,
                make_comment_prompt,
                summarize_analysis_prompt,
                supplement_info_prompt,
                refine_summary_prompt,
                summarize_record_prompt
            )
        self.choose_role_prompt = choose_role_prompt
        self.create_identity_prompt = create_identity_prompt
        self.make_comment_prompt = make_comment_prompt
        self.summarize_analysis_prompt = summarize_analysis_prompt
        self.supplement_info_prompt = supplement_info_prompt
        self.refine_summary_prompt = refine_summary_prompt
        self.summarize_record_prompt = summarize_record_prompt

    def ask_llm(self, llm, template, slot_map):
        prompt = PromptTemplate(
            input_variables=list(slot_map.keys()),
            template=template,
        )
        prompt_str = prompt.format(**slot_map)
        logger.info('\nPrompt:\n' + prompt_str)
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.apply([slot_map])[0]['text']
        logger.info('\nOutput:\n' + output)
        return output

    def choose_role(self, llm, record):
        output = self.ask_llm(llm, self.choose_role_prompt, {'record': record})
        return output

    def create_identity(self, llm, role):
        output = self.ask_llm(llm, self.create_identity_prompt, {'role': role})
        return output

    def make_comment(self, llm, record, identity):
        output = self.ask_llm(llm, self.make_comment_prompt, {'record': record, 'identity': identity})
        return output

    def summarize_analysis(self, llm, comments):
        comments_str = '\n\n'.join(comments.values())
        output = self.ask_llm(llm, self.summarize_analysis_prompt, {'comments': comments_str})
        return output

    def supplement_info(self, llm, record, identity, summary):
        output = self.ask_llm(llm, self.supplement_info_prompt, {'record': record, 'identity': identity, 'summary': summary})
        return output

    def refine_summary(self, llm, summary, supplements):
        supplement_str = '\n\n'.join(supplements.values())
        output = self.ask_llm(llm, self.refine_summary_prompt, {'summary': summary, 'supplement': supplement_str})
        return output

    def create_report(self, llm, record, summary):
        abstract = self.ask_llm(llm, self.summarize_record_prompt, {'record': record})
        if self.lang == 'zh':
            report = '病历摘要：\n' + abstract + '\n\n' + summary
        else:
            report = 'Medical record abstract: \n' + abstract + '\n\n' + summary
        return report

    def save_report(self, task_id, report):
        report_path = os.path.join(os.path.dirname(__file__), '../report', task_id+'.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

    def orchestrate(self, record):
        llm = load_llm()
        role_output = self.choose_role(llm, record)
        if utils.is_json(role_output):
            role_list = json.loads(role_output)
            identities = dict()
            comments = dict()
            for role in role_list:
                identity_output = self.create_identity(llm, role)
                identities[role] = identity_output
            for role, identity in identities.items():
                comment = self.make_comment(llm, record, identity)
                comments[role] = comment
            summary = self.summarize_analysis(llm, comments)
            ending_roles = list()
            while len(ending_roles) < len(identities):
                supplements = dict()
                for role, identity in identities.items():
                    if role in ending_roles:
                        continue
                    supple = self.supplement_info(llm, record, identity, summary)
                    if supple.strip() in ['无', 'none']:
                        ending_roles.append(role)
                    else:
                        supplements[role] = supple
                if len(supplements) > 0:
                    summary = self.refine_summary(llm, summary, supplements)
            report = self.create_report(llm, record, summary)
            self.save_report(task_id, report)
        else:
            raise ValueError('Fail to parse role list.')
