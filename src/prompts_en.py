choose_role_prompt = '''
You are a clinical doctor with extensive knowledge and experience in the field of clinical medicine.
Please output which departments the patient described in the following medical record needs to see, with no more than 5 results.
You can first analyze what aspects of the medical record have problems, and then analyze which departments are related to these problems.
Medical record:
{record}
Please strictly output the results in the following JSON format:
[
    "Department Name",
]
'''

create_identity_prompt = '''
Write a high-quality professional description starting with "You are a" for a {role} doctor, including the scope of knowledge and specific abilities. Please use you/yours and keep the response within 30 characters.
'''

make_comment_prompt = '''
{identity}
Here is a medical record:
{record}
Please provide a diagnosis, further examinations, and treatment plan within your professional field. Do not describe normal conditions, only analyze abnormal situations.
'''

summarize_analysis_prompt = '''
Here is the diagnostic report for a medical record:
{comments}
Please categorize all the above content into three sections: Diagnosis, Further Examinations, and Treatment Plan. Do not omit any information, modify the original report's meaning, or add non-existent information. Please provide the results directly without any explanations.
'''

supplement_info_prompt = '''
{identity}
Here is a medical record:
{record}
The summary of the discussion on this medical record is as follows:
{summary}
Do you have any additional comments on the diagnosis and treatment plan from your professional perspective? If you have no additional comments, please reply with "none". Otherwise, please provide your supplementary opinions.'''

refine_summary_prompt = '''
Below is the diagnosis result of a medical record:
```
{summary}
```
Below are the supplementary advice regarding the above discussion result:
```
{supplement}
```
Based on the supplementary advice above, please provide the updated diagnosis result without explanation.
'''

summarize_record_prompt = '''
Summarize the basic information and progression of the following medical record and output the results directly.
{record}
'''
