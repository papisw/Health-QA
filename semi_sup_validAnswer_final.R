options(java.parameters = "-Xmx2048m")
library(tm)
library(stringr)
library("stringdist")
library("stringi")
library(upclass)
library(e1071)
library(dtw)
library(ROCR)
library(DMwR)
library(gbm)
library("RWeka")
library(FSelector)
library(nnet)

# set directory and load data (labeled data)
setwd("/path_to/Health_care Analytics")
input<-read.csv("mydata_label_final.csv")
q_new<-as.character(input[,1]); q_new_label<-q_new
q_old<-as.character(input[,2]); q_old_label<-q_old
ans_old<-as.character(input[,3])
label<-input[,4]

# group records extracted the same question together
prev<-""
n<-1
data_q<-list(c(0))
temp<-vector()
for (i in 1:length(q_new)){
  test<-q_new[i]
  if (test==prev | i==1){
    temp<-c(temp,i)
    prev<-test
  }else if(test!=prev & i!=1){
    data_q[[n]]<-temp
    temp<-c(i)
    prev<-test
    n<-n+1
  }
}

# phase1 performance
valid_ans_count<-0
for (q in 1:length(data_q)){
  record_index<-data_q[[q]]
  label_val<-label[record_index]
  if (1 %in% label_val){
    valid_ans_count<-valid_ans_count+1
  }
}
acc_rate<-valid_ans_count/length(data_q)



# load unlabeled data
input2<-read.csv("mydata_unlabel_final.csv")
q_new2<-as.character(input2[,1])
q_old2<-as.character(input2[,2])
ans_old2<-as.character(input2[,3])

q_new<-c(q_new,q_new2); q_old<-c(q_old,q_old2); ans_old<-c(ans_old,ans_old2)

# load matching UMLS terms
drug<-read.csv("metamap_drug.txt",header=FALSE)
drug<-drug[,1]
drug<-unique(drug)

# load answers 
answer<-readLines("best_answer_out_text.txt")
rep<-readLines("rep_out_text.txt")

# conver to lowercase
q_new<-tolower(q_new)
q_old<-tolower(q_old)

# construct a list of index used for a 10-fold cross validation 
CVInd <- function(n,K) {  #n is sample size; K is number of parts; returns K-length list of indices for each part
  m<-floor(n/K)  #approximate size of each part
  r<-n-m*K  
  I<-sample(n,n)  #random reordering of the indices
  Ind<-list()  #will be list of indices for all K parts
  length(Ind)<-K
  for (k in 1:K) {
    if (k <= r) kpart <- ((m+1)*(k-1)+1):((m+1)*k)  
    else kpart<-((m+1)*r+m*(k-r-1)+1):((m+1)*r+m*(k-r))
    Ind[[k]] <- I[kpart]  #indices for kth part of data
  }
  Ind
}

# lv word distance to compute dtw
word_dist<-function(x,y){
  
  dist<-matrix(0,length(x),length(y))
  for (i in 1:length(x)){
    for (j in 1:length(y)){
      tag_error_lv<-try(stringdist(x[i],y[j],method="lv"),TRUE)
      if ('try-error' %in% class(tag_error_lv)){
        print ("error_lv")
        print (x[i])
        print (y[j])
        dist[i,j]<-1000
      }
      else{
        dist[i,j]<-stringdist(x[i],y[j],method="lv")
      }       
    }
  }
  
  dist
}

# compute feature values

# text length
textlen_new<-sapply(q_new,function(x) length(scan_tokenizer(x)))
textlen_old<-sapply(q_old,function(x) length(scan_tokenizer(x)))
# textlen_ans<-sapply(ans_old,function(x) length(scan_tokenizer(x)))

# stopwords
stw<-stopwords(kind="en")
stw<-sapply(stw, function(x) paste("",x,""))
txtnew<-sapply(q_new, function(x) paste(paste("",x,"")))
txtold<-sapply(q_old, function(x) paste(paste("",x,"")))

stw_new<-sapply(txtnew, function(x) sum(str_count(x,stw)))
stw_old<-sapply(txtold, function(x) sum(str_count(x,stw)))


# construct a document-term matrix between new quesiton and old question
subqTrain<-c(q_new,q_old); subqTrain<-unique(subqTrain)
corpus<-Corpus(VectorSource(subqTrain),list(weighting=weightTf))

corpus<-tm_map(corpus,tolower)
corpus<-tm_map(corpus,removePunctuation)
corpus<-tm_map(corpus,removeNumbers)
corpus<-tm_map(corpus,removeWords,stopwords('english'))
corpus<-tm_map(corpus,stripWhitespace)
corpus<-tm_map(corpus,PlainTextDocument)
DTMq<-DocumentTermMatrix(corpus,control=list(minWordLength=3))
DTMq_tfxidf <- weightTfIdf(DTMq,normalize=TRUE)
matQ <- as.matrix(DTMq_tfxidf)+1


combineQAnew<-c(q_new,ans_old);  combineQAnew<-unique(combineQAnew)
combineQAold<-c(q_old,ans_old);  combineQAold<-unique(combineQAold)

# construct a document-term matrix between new quesiton and old question
corpus<-Corpus(VectorSource(combineQAnew),list(weighting=weightTf))

corpus<-tm_map(corpus,tolower)
corpus<-tm_map(corpus,removePunctuation)
corpus<-tm_map(corpus,removeNumbers)
corpus<-tm_map(corpus,removeWords,stopwords('english'))
corpus<-tm_map(corpus,stripWhitespace)
corpus<-tm_map(corpus,PlainTextDocument)
DTMqanew<-DocumentTermMatrix(corpus,control=list(minWordLength=3))
DTMqanew_tfxidf <- weightTfIdf(DTMqanew,normalize=TRUE)

matQAnew <- as.matrix(DTMqanew_tfxidf)+1

# construct a document-term matrix between old quesiton and old answer
corpus<-Corpus(VectorSource(combineQAold),list(weighting=weightTf))

corpus<-tm_map(corpus,tolower)
corpus<-tm_map(corpus,removePunctuation)
corpus<-tm_map(corpus,removeNumbers)
corpus<-tm_map(corpus,removeWords,stopwords('english'))
corpus<-tm_map(corpus,stripWhitespace)
corpus<-tm_map(corpus,PlainTextDocument)
DTMqaold<-DocumentTermMatrix(corpus,control=list(minWordLength=3))
DTMqaold_tfxidf <- weightTfIdf(DTMqaold,normalize=TRUE)
matQAold <- as.matrix(DTMqaold_tfxidf)+1


distQ<-vector();  distQAnew<-vector(); distQAold<-vector();  distQAdiff<-vector();  
distQDTW<-vector();  distQAnewDTW<-vector();   distQAoldDTW<-vector();   distQAdiffDTW<-vector()
overlap_q<-vector(); overlap_a<-vector()
tag_covering<-vector();  
setdiff_q<-vector(); setdiff_a<-vector()

for (i in 1:length(q_old)){ 
  print (i)
  
  # compute VS-based distance 
  index_qold<-which(subqTrain==q_old[i]); index_qnew<-which(subqTrain==q_new[i])
  x1<-matQ[index_qold,]; x2<-matQ[index_qnew,]
  distQ[i]<-dist(rbind(x1,x2),"euclidean") 
  
  index_qnew<-which(combineQAnew==q_new[i]); index_aold<-which(combineQAnew==ans_old[i])
  x1<-matQAnew[index_qnew,]; x2<-matQAnew[index_aold,]
  distQAnew[i]<-dist(rbind(x1,x2),"euclidean")
  
  index_qold<-which(combineQAold==q_old[i]); index_aold<-which(combineQAold==ans_old[i])
  x1<-matQAold[index_qold,]; x2<-matQAold[index_aold,]
  distQAold[i]<-dist(rbind(x1,x2),"euclidean")
  distQAdiff[i]<-abs(distQAnew[i]-distQAold[i])
  
  # compute DTW-based distance
  Qnew<-removeWords(q_new[i],stopwords("english")); Qold<-removeWords(q_old[i],stopwords("english")); ans<-removeWords(ans_old[i],stopwords("english"))
  word_dist_matrix<-word_dist(Qnew,Qold)
  distQDTW[i]<-dtw(word_dist_matrix)$normalizedDistance
  word_dist_matrix<-word_dist(Qnew,ans)
  distQAnewDTW[i]<-dtw(word_dist_matrix)$normalizedDistance
  word_dist_matrix<-word_dist(Qold,ans)
  distQAoldDTW[i]<-dtw(word_dist_matrix)$normalizedDistance
  distQAdiffDTW[i]<-abs(distQAnewDTW[i]-distQAoldDTW[i])
  
  # matching UMLS terms
  result_match<-sapply(drug,function(x) grepl(x,q_old[i]))
  term_match_qold<-drug[which(result_match==TRUE)]
  result_match<-sapply(drug,function(x) grepl(x,q_new[i]))
  term_match_qnew<-drug[which(result_match==TRUE)]
  result_match<-sapply(drug,function(x) grepl(x,ans_old[i]))
  term_match_ansold<-drug[which(result_match==TRUE)]
  
  overlap_q[i]<-length(intersect(term_match_qold,term_match_qnew))
  overlap_a[i]<-length(intersect(term_match_ansold,term_match_qnew))
  
  targetcover<-setdiff(term_match_qnew,term_match_qold)
  if (length(targetcover)!=0){
    tag_covering[i]<-"0"
  }else{
    tag_covering[i]<-"1"
  }
  
  setdiff_q[i]<-length(c(setdiff(term_match_qold,term_match_qnew),setdiff(term_match_qnew,term_match_qold)))
  setdiff_a[i]<-length(c(setdiff(term_match_ansold,term_match_qnew),setdiff(term_match_qnew,term_match_ansold)))
  
}

# collect and store all calculated feature values 
# all features including health features
traindata<-vector()
traindata<-cbind(traindata,textlen_new,textlen_old,stw_new,stw_old,distQ,distQAdiff,distQDTW,distQAdiffDTW,overlap_q,overlap_a,tag_covering,setdiff_q,setdiff_a)

# no health features
#traindata<-cbind(traindata,textlen_new,textlen_old,stw_new,stw_old,distQ,distQAdiff,distQDTW,distQAdiffDTW)


class(traindata)<-"numeric"
x<-as.matrix(traindata)
x[,1:ncol(x)]<-scale(x[,1:ncol(x)],scale=FALSE) # scale value

# clean na value
error<-vector()
tagerr<-is.na(x)
for (i in 1:ncol(x)){
  indexerr<-which(tagerr[,i]==TRUE)
  if (length(indexerr)>0){
    error<-c(error,indexerr)
  }
}
error<-unique(error)
x<-x[-error,]

# error2<-error[error<=length(label)]
# label<-label[-error2]
# orig_dat<-cbind(q_new,q_old,ans_old)
# orig_dat<-orig_dat[-error,]



# supervised/ semi-supervised models

p<-0   
Nrep<-1
perf_record<-matrix(0,Nrep,6)
perf_q<-matrix(0,Nrep,5)
for (n in 1:Nrep){
  ypred<-label
  weak<-vector(); medium<-vector(); strong<-vector(); all<-vector() # measure performance (question-based)
  sum_rank<-0; sum_q<-0 # measure performance (question-based)
  IndQ<-CVInd(n=length(data_q),10)
  KK<-length(IndQ)
  for (kk in 1:KK){
    # question index for test set
    listQtest<-IndQ[[kk]]
    
    # collect record index for test set
    test_range<-vector(); train_range<-1:length(label)
    for (q in listQtest){
      test_range<-c(test_range,data_q[[q]])
    }
    
    train_range<-train_range[-test_range]
    data<-cbind(x[1:length(label),],label)
    label_feature<- data[train_range,-ncol(data)]; label_class<-label[train_range]; ytrue= label[test_range]
    temp_index<-length(label)+1
    unlabel_feature<-x[temp_index:nrow(x),]
    
    col<-colnames(data)
    train<-cbind(label_feature,as.factor(label_class))
    colnames(train)<-col
    
    train<-as.data.frame(label_feature)
    train<-data.frame(train,"label"=as.factor(label_class))
   
    # re-sampling data addressing unbalanced issue
#     newdata<-as.data.frame(train) 
    newdata<-SMOTE(label~.,as.data.frame(train),perc.over=100,perc.under=200)
    colnames(newdata)<-col
    
    
    # naive bayes
#       m<-naiveBayes(label~.,data=as.data.frame(newdata),laplace=5)
#       prob<-predict(m,as.data.frame(unlabel_feature),type ="raw")
#       phat<-prob[,2]
    
    #SVM
#     m<-svm(label~.,newdata,kernel="radial",gamma=0.001,cost=10,probability=TRUE)
#     preds<-predict(m,as.data.frame(unlabel_feature),probability=TRUE)
#     prob<-attr(preds,"probabilities")
#     phat<-prob[,2]
    
    # nnet
    m<-nnet(label~.,newdata,linout=F,skip=F,size=5,decay=0.01,maxit=1000,trace=F,na.rm=TRUE)
    test_dat<-as.data.frame(unlabel_feature); row<-1:nrow(test_dat); rownames(test_dat)<-row
    prob<-predict(m,as.data.frame(test_dat),type ="raw")
    phat<-prob[,1]
    
    # assign label with specified cut-off probability value
    lab<-as.numeric(phat>=0.75)    
    unlabel<-cbind(unlabel_feature,lab)
    colnames(unlabel)<-col
    
        # repeat EM algorithm
        for (i in 1:10){
          Ind<-CVInd(n=nrow(unlabel),10)
          K<-length(Ind)
          for (k in 1:K){
            new_train<-rbind(unlabel[Ind[[k]],],train)
            lab<-new_train$label
            new_train<-new_train[,-ncol(new_train)]
            new_train<-data.frame(new_train,"label"=as.factor(lab))
#             newdata<-as.data.frame(new_train)
            newdata<-SMOTE(label~.,as.data.frame(new_train),perc.over=100,perc.under=200)
            
#                     # naive bayes
#                     m<-naiveBayes(label~.,data=as.data.frame(newdata))
#                     prob<-predict(m,as.data.frame(unlabel[-Ind[[k]],-ncol(unlabel)]),type ="raw")
#                     phat<-prob[,2]
            
                     #SVM
#                     m<-svm(label~.,newdata,kernel="radial",gamma=0.001,cost=10,probability=TRUE)
#                     preds<-predict(m,as.data.frame(unlabel[-Ind[[k]],-ncol(unlabel)]),probability=TRUE)
#                     prob<-attr(preds,"probabilities")
#                     phat<-prob[,2]
            
            m<-nnet(label~.,newdata,linout=F,skip=F,size=5,decay=0.01,maxit=1000,trace=F,na.rm=TRUE)
            test_dat<-as.data.frame(unlabel[-Ind[[k]],-ncol(unlabel)]); row<-1:nrow(test_dat); rownames(test_dat)<-row
            prob<-predict(m,as.data.frame(test_dat),type ="raw")
            phat<-prob[,1]
            
            unlabel[-Ind[[k]],ncol(unlabel)]<-as.numeric(phat>=0.75)
          }
        }
   
        # conbine unlabel with label
        final_train<-rbind(unlabel,train)
        lab<-final_train$label
        final_train<-as.data.frame(final_train); 
        final_train<-data.frame(final_train[,-ncol(final_train)],label=as.factor(lab))
        
        newdata<-SMOTE(label~.,as.data.frame(final_train),perc.over=100,perc.under=200)
#         newdata<-as.data.frame(final_train)
        colnames(newdata)<-col
    
#     #naive bayes
#         m<-naiveBayes(label~.,data=as.data.frame(final_train))
#         prob<-predict(m,as.data.frame(x[test_range,]),type="raw")
#         phat<-prob[,2]
    
    #SVM
#         m<-svm(label~.,data=newdata,kernel="radial",gamma=0.001,cost=10,probability=TRUE)
#         preds<-predict(m,as.data.frame(x[test_range,]),probability=TRUE)
#         prob<-attr(preds,"probabilities")
#         phat<-prob[,2]
    
    # nnet
        m<-nnet(label~.,newdata,linout=F,skip=F,size=5,decay=0.01,maxit=1000,trace=F)
        test_dat<-as.data.frame(x[test_range,]); row<-1:nrow(test_dat); rownames(test_dat)<-row
        prob<-predict(m,as.data.frame(test_dat),type ="raw") 
        phat<-prob[,1]
    
    class<-as.numeric(phat>=0.75)
    
    

    # question performance
    sum<-0
    for (q in listQtest){
      record_index<-data_q[[q]]
      len_index<-length(data_q[[q]])
      temp1<-sum+1; temp2<-sum+len_index
      count1_label<-sum(ytrue[temp1:temp2]==1)
      count0_label<-sum(ytrue[temp1:temp2]==0)
      rank_index<-rank(-phat[temp1:temp2]) # rank from largest to smallest
      count<-0
      count_0<-0
      order<-vector()
      for (l in 1:len_index){
        index<-l+sum
        if (class[index]==1 & ytrue[index]==1)
          count<-count+1
        if (class[index]==0 & ytrue[index]==0)
          count_0<-count_0+1
        
        #rank
        if (ytrue[index]==1){
          order<-c(order,which(rank_index==l))
        }
      }
      if (count1_label>0){
        sum_rank<-sum_rank+ 1/min(order)
        sum_q<-sum_q+1
      }
      
      if (count_0 < ceiling(p*count0_label)){
        weak<-c(weak,0); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
      }else{
        if (count1_label==0 & count==0){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,1); all<-c(all,1)
        }else if(count1_label==0 & count!=0){
          weak<-c(weak,0); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label==1 & count>=1){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,1); all<-c(all,1)
        }else if(count1_label==1 & count<1){
          weak<-c(weak,0); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label==2 & count>=2){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,1); all<-c(all,1)
        }else if(count1_label==2 & count==1){
          weak<-c(weak,1); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label==2 & count<1){
          weak<-c(weak,0); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label==3 & count>=3){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,1); all<-c(all,1)
        }else if(count1_label==3 & count==2){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label==3 & count==1){
          weak<-c(weak,1); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label==3 & count<1){
          weak<-c(weak,0); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label>3 & count==count1_label){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,1); all<-c(all,1)
        }else if(count1_label>3 & count==3){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,1); all<-c(all,0)
        }else if(count1_label>3 & count==2){
          weak<-c(weak,1); medium<-c(medium,1); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label>3 & count==1){
          weak<-c(weak,1); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }else if(count1_label>3 & count<1){
          weak<-c(weak,0); medium<-c(medium,0); strong<-c(strong,0); all<-c(all,0)
        }
      
      }
      
      sum<-sum+len_index
    }
    ypred[test_range]<-class
  }
  # record performance
  pre<-prediction(ypred,as.numeric(label))
  perf<-performance(pre,"auc")
  auc<-attributes(perf)$y.values[[1]]
  TP<-sum(label==1 & ypred==1); TN<-sum(label==0 & ypred==0); FP<-sum(label==0 & ypred==1); FN<-sum(label==1 & ypred==0)
  f1<-2*TP/ (2*TP+FN+FP)
  perf_record[n,]<-c(auc,TP,TN,FP,FN,f1)
  
  # question performance
  weak_rate<-sum(weak==1)/length(weak); medium_rate<-sum(medium==1)/length(medium); strong_rate<-sum(strong==1)/length(strong); all_rate<-sum(all==1)/length(all)
  rank_rate<-sum_rank/sum_q
  perf_q[n,]<-c(weak_rate,medium_rate,strong_rate,all_rate,rank_rate)
}


a<-apply(perf_record,2,mean)
prec<-a[2]/(a[2]+a[4]); rec<-a[2]/(a[2]+a[5])
print (a)
print (prec)
print (rec)
apply(perf_q,2,mean)


