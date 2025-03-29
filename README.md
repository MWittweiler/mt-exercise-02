# MT Exercise 2: Pytorch RNN Language Models

MT 2025 – Exercise 2 – Michael Wittweiler 16-727-992
https://github.com/MWittweiler/mt-exercise-02 

# Task 1

Modifications:
(Obviously, the renaming of the folder from grimm to bible whereever necessary)

Download_data.sh
-	I had to manually load NLTK punkt_tab in python3 in order for the script to work
o	>>> import nltk
o	>>> nltk.download("punkt_tab")

preprocess.py
-	Added two lines that remove the enumeration of Bible verses

main.py
-	Change line 246 to: model = torch.load(f, weights_only=False)
o	Otherwise I ran into an error

generate.py
-	Same as in main.py, added weights_only=False



# TASK 2

# Step-by-Step instructions:
- git clone https://github.com/MWittweiler/mt-exercise-02
- cd mt-exercise-02
- ./scripts/make_virtualenv.sh
- source venvs/torch3/bin/activate./scripts/install_packages.sh # I added pandas and matplotlib
- make sure that you use the MODIFIED «main.py» and «generate.py» in \tools\pytorch-examples\word_language_model\ provided in my repository on Github (they might get overwritten when executing install_packages.sh)
- ./scripts/download_data.sh
- make sure the directories «logs» and «models» as direct subfolders of «mt-exercise-02» exist and are empty as not to run into an error when saving the models. 
- ./scripts/train.sh
- ./scripts/visualize.py
- (./scripts/generate.sh (change model name if want a specific one))

If you ever stop training and start the whole process again, make sure that the log files are empty before restarting training otherwise the log files have multiple entries!

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`



