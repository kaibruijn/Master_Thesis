from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

def main():
    data = pd.read_excel("../PAN-15/logregexcel_PAN-15trainlargeconcatenated.xlsx", names=["Folder", "labels", "Text1", "Text2"])
    train_df = transform_data(data)
    
    data = pd.read_excel("../PAN-15/logregexcel_PAN-15testlargeconcatenated.xlsx", names=["Folder", "labels", "Text1", "Text2"])
    eval_df = transform_data(data)
    
    transformer(train_df, eval_df)

def transform_data(data):
    data = data[["labels", "Text1", "Text2"]]
    data["text"] = data["Text1"].map(str) +"\n\n\n"+ data["Text2"]
    data = data[1:]
    data = data[["labels", "text"]]
    transformer_dict = {-1: 0, 1: 1}
    data['labels'] = data['labels'].apply(lambda x: transformer_dict[x])
    #data['text'] = data['text'].apply(lambda x: TreebankWordDetokenizer().detokenize(x.split()))
    print(data)

    return data

def transformer(train_df, eval_df):
    model = ClassificationModel("bert", "bert-base-dutch-cased", use_cuda=False, args={"overwrite_output_dir": True})
    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df, cr=classification_report, cm=confusion_matrix)

    print(model_outputs)
    for i in model_outputs:
    	print(i)
    print(result['cr']) # Classification report
    print(result['cm']) # Confusion matrix


if __name__ == '__main__':
    main()