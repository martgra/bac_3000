#!/bin/bash
for q in Agaricus_bisporus Amanita_muscaria Amanita_virosa  Boletus_edulis Cantharellus_cibarius Cortinarius_rubellus Craterellus_cornucopioides Lactarius_deterrimus Suillus_variegatus;

do
       mkdir /home/jason/train_3/train/$q
       mkdir /home/jason/train_3/validate/$q
       for file in $(ls -p /home/jason/train_data4/$q | grep -v / | tail -400)
       do
       mv /home/jason/train_data4/$q/$file /home/jason/train_3/validate/$q;
       done

       for f in $(ls -p /home/jason/train_data4/$q);
       do mv /home/jason/train_data4/$q/$f /home/jason/train_3/train/$q;
       done
done
