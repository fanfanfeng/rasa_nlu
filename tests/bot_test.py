# create by fanfan on 2019/3/15 0015
from rasa.nlu.model import Interpreter



def Load_NLU():
    interpreter = Interpreter.load(r'E:\git-project\rasa_nlu\rasa\nlu\tmp\models\default\classify')
    return  interpreter






if __name__ == '__main__':
    interpreter = Load_NLU()
    while True:
        try:
            text = input()
            if not text == None:
                res = interpreter.parse(text)
                print(res)
        except Exception as e:
            print(e)



