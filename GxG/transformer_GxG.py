from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

def main():

    Xtrain = []
    Ytrain = []
    with open("trainGxG/GxG_News.txt") as txt:
        for line in txt:
            if line[0:8] == "<doc id=":
                Ytrain.append(line.split()[3][8])
                string=[]
            elif line[0:6] == "</doc>":
                Xtrain.append(" ".join(string))
            else:
                string.append(line)

    Xtest = []
    with open("testGxG/GxG_News.txt") as txt:
        for line in txt:
            if line[0:8] == "<doc id=":
                string=[]
            elif "</doc>" in line:
                Xtest.append(" ".join(string))
            else:
                string.append(line)

    Ytest = []
    with open("testGxG/GxG_News_gold.txt") as text:
        for line in text:
            Ytest.append(line.split()[1])

    train_df = pd.DataFrame(list(zip(Ytrain, Xtrain)), 
               columns =["labels", "text"])

    transformer_dict = {"F": 0, "M": 1}
    train_df['labels'] = train_df['labels'].apply(lambda x: transformer_dict[x])


    eval_df = pd.DataFrame(list(zip(Ytest, Xtest)), 
               columns =["labels", "text"])

    transformer_dict = {"F": 0, "M": 1}
    eval_df['labels'] = eval_df['labels'].apply(lambda x: transformer_dict[x])

    transformer(train_df, eval_df)


def transformer(train_df, eval_df):
    model = ClassificationModel("bert", "bert-base-dutch-cased", use_cuda=False, args={"overwrite_output_dir": True, "num_train_epochs": 1})
    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df, cr=classification_report, cm=confusion_matrix)

    print(model_outputs)
    print(result['cr']) # Classification report
    print(result['cm']) # Confusion matrix


if __name__ == '__main__':
    main()