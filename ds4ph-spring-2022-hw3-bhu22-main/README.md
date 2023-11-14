# HW3

For this homework, please return a single ipynb file.

* Download the Kirb21 data [here](https://raw.githubusercontent.com/smart-stats/ds4bio_book/main/book/assetts/kirby21AllLevels.csv). Pull out subject with `rawid` equal to `kirby906a_ax.img`. Do the following.
   + Calculate and report the intracranial volume (ICV) and the total brain volume (TBV)
   + Create a dataframe that has the sum of all of the regions for every type and level. So, it should have three columns, type, level and total_volume. 
   + Creata a plots to visualize this subject's Type 1 Level 2 data and Type 1 Level 3 data.
* Read in the class data from here [https://github.com/bcaffo/ds4bme/blob/master/data/classInterests.txt](https://github.com/bcaffo/ds4bme/blob/master/data/classInterests.txt). Create bar plots of year and program. 
* Download the gene expression dataset from [here](https://github.com/jhu-advdatasci/2018/blob/master/data/GSE5859_exprs.csv). Information about the data can be found in the sample info [here](https://github.com/jhu-advdatasci/2018/blob/master/data/GSE5859_sampleinfo.csv). From the gene expression data matrix, subtract the rowmean from each row and the column mean from each column so that you have a new matrix that has been demeaned across both rows and columns. Divide each column by its standard deviation.
* Download the data from [here](https://raw.githubusercontent.com/jhu-advdatasci/2018/master/data/KFF/healthcare-spending.csv). Create plots of healthcare spending versus time color coded by states.
* Refer to the previous data. Create a barplot of average health care spending by state.
