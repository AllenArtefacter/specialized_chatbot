# Specialized Chatbot
This project is about a GPT chatbot answer the question only based on the documentation we provide. It is useful when there is a need to make a internal documents responser rather than a bot will answer with some content we do not expect.

## Requirements
`llama-index`>=0.4.28  
`langchain`>=0.0.112  
`openai`==0.27.2  
`tensorflow`


## Installalation
```bash
git clone https://github.com/AllenArtefacter/specialized_chatbot.git
cd specialized_chatbot
python3 setup.py install
```


## Usage
```python
from specialized_chatbot import chatbot
bot = chatbot.Chatbot('<your document file directory>')

print(bot.continue_conversation('How are you?'))
```
