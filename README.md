# 3rd place solution WiDS Datathon 2020

Congrats to the winner team and to Oleg's Team for the 2nd spot

Our solution is mostly based on :

removing some features : [ 'hospital_id',' icu_id','ethnicity','gender', 'patient_id','encounter_id','hospital_death', **'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob' **]
cleansing some data (like when min > max)
creating few features ( top features was train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0] )
adding some pseudo labelling (10-20%)
we then blend 3 models LGB with categorical data handling, LGB with OHE and NN with weight(0.6, 0.2, 0.2)

Repeat Stratified Kfold was used and gave CV 0.129 for the blend without pseudo labelling.

I have open shared the NN kernel that doesn't really gave good result by itself (best public LB by blending seed was 0.91 , private 0.90777 ) but was adding to the blend, I ll be interested to get some feedback about NN on this dataset .

https://www.kaggle.com/jayjay75/3rd-place-nn-wids2020?scriptVersionId=29209297

./