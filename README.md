# Master_Thesis
The Genetical Influence on Writing Syle

Train and predict with GLAD:
"python3 glad-main.py --training [train_dir] --test [test_dir]"

Evaluate GLAD:
"python3 evaluation.py -t [test_dir]/truth.txt"

Information on Twin-20 predictions:
no corrections: "python3 extra_info.py"
age correction: "python3 extra_info_age.py"
gender correction: "python3 extra_info_gender.py"
age and gender correction: "python3 extra_info_age_and_gender.py"

Feature influence (change features in feature_analysis.py):
python3 feature_analysis.py --training RedSet-20/RedSet-20_all/ --test Twin-20/
cd info
python3 extra_info_features.py

All other python files can be run with "python3 [python_file]".

Version requirements:
- scikit-learn==0.17
- scipy==0.19.0

Other requirements depend on the python file, for example simpletransformers and spacy. They are not needed for the main research topic but were used for experiments.
