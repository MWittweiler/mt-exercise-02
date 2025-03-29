#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/grimm

mkdir -p $data/grimm/raw

wget https://www.gutenberg.org/cache/epub/1581/pg1581.txt
mv pg1581.txt $data/bible/raw/bible.txt

# preprocess slightly

cat $data/bible/raw/bible.txt | python $base/scripts/preprocess_raw.py > $data/bible/raw/bible.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/bible/raw/bible.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/bible/raw/bible.preprocessed.txt

# split into train, valid and test

head -n 5050 $data/bible/raw/bible.preprocessed.txt | tail -n 5000 > $data/bible/valid.txt
head -n 10050 $data/bible/raw/bible.preprocessed.txt | tail -n 5000 > $data/bible/test.txt
tail -n 40100 $data/bible/raw/bible.preprocessed.txt | head -n 40000 > $data/bible/train.txt
