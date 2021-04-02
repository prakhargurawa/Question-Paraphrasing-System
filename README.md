# Question-Paraphrasing-System
The task is to generate paraphrased questions, that is questions that have the same meaning but are different in terms of the vocabulary and grammar of the sentence  The training data is derived from the Datasets of Paraphrased SQuAD Questions https://github.com/nusnlp/paraphrasing-squad and consists of 1,118 paraphrased questions. The evaluation consists of 100 questions also derived from the Stanford Question Answering Dataset https://rajpurkar.github.io/SQuAD-explorer/

## Original Compitition Link : 
https://competitions.codalab.org/competitions/28529 

## Organizers :
John P. McCrae - Data Science Institute, National University of Ireland Galway

## Evaulation Criteria:
Evaluation will be in terms of BLEU and PINC. BLEU score measures the similarity of the paraphrases with the reference sentences. 

## Model

#### Model 1
Its a pretrained T5 transformer which is trained on a huge quora dataset and give good results :
Harmonic Mean : 0.118 (1)	
BLUE : 0.096 (2)	
PINC : 0.586 (1)

#### Model 2
Trained T5 transformer on my dataset on google colab and then downloaded those trained models to find paraphrases on eval
Here i have experimented first by splitting the train dataset in 80:20 and then in 95:5 as the dataset is small, the second gives better results

With 80:20 split :
Harmonic Mean : 0.085 (1)
BLUE : 0.104 (2)
PINC : 0.322 (1)

With 95:5 split:
Harmonic Mean : 0.104 (1)
BLUE : 0.113 (2)
PINC : 0.378 (1)

#### Model 3
Tried creating a systrem from scratch , used a encoder decoder model. Created two models here where the second one a stacked LSTM enoder decoder. I thought increasing the complexity will help me getting good results but results are unsatisfactory. So I have not submitted these in compitition

#### Model 4
Tried experimenting with BART for paraphrasing task but results are not good. BART mimics the input sequence as is it which is also one of a shortcoming of BART.
Both in Model 3 and Model 4 I felt working with a bigger corpus can give some great results 
Other dataset for paraphrasing can be found here:

* https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs
* https://github.com/google-research-datasets/paws#paws-wiki




