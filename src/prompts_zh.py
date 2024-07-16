choose_role_prompt = '''
你是一位临床医生，在临床医学领域有丰富的知识和经验。
请输出以下病历里描述的患者需要看哪些科室，结果不超过5个。
你可以先分析病历中有哪些方面的问题，再分析这些问题分别和哪个科室相关。
病历：
{record}
严格按照以下json格式输出结果：
[
    "科室名称",
]
'''

create_identity_prompt = '''
以"你是一位"开头为一个{role}医生写一段高质量的身份描述，包括知识范围和具体能力，使用第二人称，结果控制在30字以内。
'''

make_comment_prompt = '''
{identity}
下面是一份病历：
{record}
请在你的专业领域内输出诊断、进一步检查、治疗方案。不要描述正常情况，仅分析异常情况。
'''

summarize_analysis_prompt = '''
下面是对一份病历的诊疗报告：
{comments}
把上面的所有内容分类汇总为诊断、进一步检查、治疗方案三部分，不要遗漏信息，不要修改原始报告的意思，不要添加不存在的信息。直接输出结果，不要解释。
'''

supplement_info_prompt = '''
{identity}
下面是一份病历：
{record}
对这份病历的讨论总结如下：
{summary}
从你的专业角度对诊疗方案是否还有补充意见，如果没有补充意见请回复“无”，否则输出补充意见。
'''

refine_summary_prompt = '''
下面是一份病历的讨论结果:
```
{summary}
```
下面是针对上面讨论结果的补充建议：
```
{supplement}
```
根据上面的补充建议补充原始讨论结果，输出新的结果，不要解释。
'''

summarize_record_prompt = '''
总结下面病历里的基本情况和病情进展，直接输出结果。
{record}
'''
