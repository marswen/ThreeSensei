import fire
from steps import ThreeSensei


def main(file_path, lang='zh'):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    three_sensei = ThreeSensei(lang)
    three_sensei.orchestrate(text)


if __name__ == '__main__':
    fire.Fire(main)
