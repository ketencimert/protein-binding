# Machine Learning for Functional Genomics - Final Project - mk4139

Data is available at:

https://drive.google.com/drive/folders/1_2WmaUKLWY2q7kN4TLkreHFvZBoZv8dS

Train the model on ChIP-Seq dataset:

```
python main.py --num_motifs 1 --kernel_size 20 --epochs 400 --device [your device] --dataname chip --datadir [your datadir]
```

Train the model on ATAC-Seq dataset:

```
python main.py --num_motifs 160 --kernel_size 20 --epochs 400 --device [your device] --dataname atac --datadir [your datadir]
```

