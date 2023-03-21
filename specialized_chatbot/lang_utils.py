import langid
import json
import os

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data','lang_code.json'), 'r') as f:
    lang_code = json.load(f)

def lang_detect(text:str)->str:
    """detect language
    Parameter:
    --------
        text: str
            the text for which the langudate you want to detect

    Return
    ------
        langudge
    """
    # Language detection
    try:
        code = langid.classify(text)[0]
        lang = lang_code[code]
    except Exception as e:
        print(e)
        lang = 'English'
    return lang
