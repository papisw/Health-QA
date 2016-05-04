library(tm)
library("RWeka")
library("topicmodels")
library("stringdist")
library("stringi")
library("dtw")

# specifiy directory and load data
setwd("/path_to/Health-QA/data")
question_input<-read.csv("subQ_final.csv")

# store sub-question
question<-vector()
for (i in 1:nrow(question_input)){
  num<-sum(question_input[i,]!="")
  q=""
  for (j in 1:num){
    q<-paste(q,question_input[i,j],"#",sep="")
  }
  question[i]<-q
}

# load data
full_question<-readLines("qfile_final.txt"); full_question<-gsub("\t"," ",full_question)
answer<-readLines("best_answer_out_text.txt")
rep_answer<-readLines("rep_out_text.txt")
question<-tolower(question); full_question<-tolower(full_question); answer<-tolower(answer); rep_answer<-tolower(rep_answer)
drug<-read.csv("metamap_drug.txt",header=FALSE)
drug<-drug[,1]
drug<-unique(drug)

weight<-0.5

# construct a cross-distance matrix
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

# compute a distance between 2 strings
StringMatch <- function(test,trainQ,answer,drug,weight,method){
  # test = tested subQuestion , trainQ = set of other subQuestion
  dist<-vector()
  
  # do tf-idf
  input<-c(test,trainQ)
  corpus<-Corpus(VectorSource(input),list(weighting=weightTf))
  corpus<-tm_map(corpus,removePunctuation)
  corpus<-tm_map(corpus,removeNumbers)
  corpus<-tm_map(corpus,stripWhitespace)
  dtm<-DocumentTermMatrix(corpus)
  dtm_tfxidf <- weightTfIdf(dtm,normalize=TRUE)
  mat <- as.matrix(dtm_tfxidf)
  
  for (i in 1:length(trainQ)){
    train<-trainQ[i]
    test<-removeWords(test,stopwords("english")); train<-removeWords(train,stopwords("english")) 
    train_token<-scan_tokenizer(train); test_token<-scan_tokenizer(test); ans_token<-scan_tokenizer(answer)
    
    x<-gsub(" ","",train_token); y<-gsub(" ","",test_token)  #remove space
    x<-gsub("\\s","",x); y<-gsub("\\s","",y) #remove whitespace
    x<-gsub("[[:punct:]]","",x); y<-gsub("[[:punct:]]","",y); #remove punctuation
    x<-gsub("\\d","",x); y<-gsub("\\d","",y) #remove number
    
    if (method=="lv"){
      word_dist_matrix<-word_dist(test,train)
      sentence_dist<-dtw(word_dist_matrix)$normalizedDistance
    }
    else{
      sentence_dist<-dist(rbind(mat[1,],mat[i+1,]),"euclidean")
#       x1<-mat[1,]; x2<-mat[i+1,]
#       sentence_dist<-(x1%*%x2 / sqrt(x1%*%x1 * x2%*%x2))[1,1]
    }
    
    # check whether sentences contain matching UMLS topics
    result_match<-sapply(drug,function(x) grepl(x,test))
    term_match<-drug[which(result_match==TRUE)]
    skip=0
    if (length(which(result_match==TRUE))==0){
      skip=1
    }
    if (skip==0){
      for (c in 1:length(term_match)){
        tag_error_tr<-try(grepl(term_match[c],train),TRUE)
        if ('try-error' %in% class(tag_error_tr)){
          print (term_match[c])
          tag1_q<-FALSE
        }else{
          tag1_q<-grepl(term_match[c],train)
        }
        
        if (tag1_q==TRUE){
          sentence_dist = sentence_dist*weight # multiply computed distance with weight if matching occurs
        }
      }
      
    }
    
    dist<-c(dist,sentence_dist) 
  }
  dist
}


#random number to be test case
num<-5
test<-sample(1:4216,num,replace=F)
Qtrain<-question[-test]
Atrain<-answer[-test]
Reptrain<-rep_answer[-test]
sink('test_out.txt')

# question words 
qword_type = c("when","what time","why","what reason","how long","how much","how many","how often","how frequently","where","who","what","which","whose","whom")
aux_verb = c("shall","should","will","would","can","could","may","might","do","does","is","am","are","has","have")
Q_regex_list = c()
Q_regex=""
aux_regex = ""
for (i in 1:length(aux_verb)){
  if (aux_regex==""){
    aux_regex = aux_verb[i]
  }else{
    aux_regex = paste(aux_regex,aux_verb[i],sep="|")
  }
  for (j in 1:length(qword_type)){
    word = paste(qword_type[j],aux_verb[i])
    Q_regex_list = c(Q_regex_list,word)
    if (Q_regex==""){
      Q_regex = word
    }
    else{
      Q_regex = paste(Q_regex,word,sep="|")
    }
    
  }
}
aux_regex = paste("^(",aux_regex,")",sep="")

# cases do not contain question words
indexnoq<-vector()

for (i in 1:length(Qtrain)){
  qword<-""
  detect_regex<-stri_detect_regex(Qtrain[i],Q_regex)
  detect_aux<-stri_detect_regex(Qtrain[i],aux_regex)
  if (length(which(detect_aux==TRUE))!=0){
    qword<-"YesNo"
  } 
  detect2<-stri_detect_fixed(Qtrain[i],qword_type)
  if (length(which(detect2==TRUE))==0 && qword!="YesNo"){
    qword <- "none"
  }
  if (qword=="none"){
    indexnoq<-c(indexnoq,i)
  }
}


for (i in 1:num){
  
  index<-test[i]; test_q<-question[index]
  cat(sprintf("full question: %s\n",full_question[index]))
  cat(sprintf("question: %s\n",question[index]))
  list_q<-unlist(strsplit(test_q,"#",fixed=TRUE))
  row<-vector()
  
  for (j in 1:length(list_q)){
    qword<-""
    data_out<-vector()
    cat(sprintf("sub_question: %d\n",j))
    detect_regex<-stri_detect_regex(list_q[j],Q_regex)
    detect_aux<-stri_detect_regex(list_q[i],aux_regex)
    if (length(which(detect_aux==TRUE))!=0){
      qword<-"YesNo"
    } 
    detect2<-stri_detect_fixed(list_q[j],qword_type)
    if (length(which(detect2==TRUE))==0 && qword!="YesNo"){
      qword <- "none"
    }else {
      indexQ = which(detect2==TRUE)
      if (indexQ == 1 || indexQ ==2){
        qword<-"when"
      } else if (indexQ ==3 || indexQ ==4){
        qword<- "why"
      } else if (indexQ ==5 || indexQ ==6 || indexQ==7){
        qword<- "quantity"
      } else if (indexQ ==8 || indexQ ==9){
        qword<- "frequency"
      } else{
        qword<-qword_type[which(detect2==TRUE)]
      }
      
    }  
    
    for (q in 1:length(qword)){
      Qmatch<-vector(); Amatch<-vector(); Repmatch<-vector()
      if (qword[q]=="none"){
        Qtrain2<-Qtrain[indexnoq]; Atrain2<-Atrain[indexnoq]; Reptrain2<-Reptrain[indexnoq]
        for (qt in 1:length(Qtrain2)){
          subQtrain = unlist(strsplit(Qtrain2[qt],"#",fixed=TRUE))
          for (s in 1:length(subQtrain)){
            Qmatch<-c(Qmatch,subQtrain[s])
            Amatch<-c(Amatch,Atrain2[qt])
            Repmatch<-c(Repmatch,Reptrain2[qt])
          }
        }
      } else{
        index_match = which(stri_detect_fixed(Qtrain,qword[q])==TRUE)
        Qtrain2<-Qtrain[index_match]; Atrain2<-Atrain[index_match]; Reptrain2<-Reptrain[index_match]
        for (qt in 1:length(Qtrain2)){
          subQtrain = unlist(strsplit(Qtrain2[qt],"#",fixed=TRUE))
          for (s in 1:length(subQtrain)){
            if (length(grep(qword[q],subQtrain[s]))>0){
              Qmatch<-c(Qmatch,subQtrain[s])
              Amatch<-c(Amatch,Atrain2[qt])
              Repmatch<-c(Repmatch,Reptrain2[qt])
            }
          }
        }
      }
      
      
      # lv method
      
      list_dist<-StringMatch(list_q[j],Qmatch,Amatch,drug,weight,"lv")
      lowest_dist<-head(sort(list_dist),2)
      ans_tag<-unlist(sapply(lowest_dist,function(x) which(x==list_dist)))
      ans<-Amatch[ans_tag]
      ans2<-Repmatch[ans_tag]
      matched_question<-Qmatch[ans_tag]
      dist<-list_dist[ans_tag]
      cat(sprintf("method: Lavenshtien\n"))
      for (rep in 1:2){
        
        cat(sprintf("Top answer: %d\n",rep))
        cat(sprintf("matched question: %s\n",matched_question[rep]))
        cat(sprintf("answer: %s\n",ans[rep]))
        t1<-gsub(",","+",list_q[j]); t2<-gsub(",","+",matched_question[rep]); t3<-gsub(",","+",ans[rep]);
        data_out<-rbind(data_out,c(t1,t2,t3))
        
        cat(sprintf("2nd answer: %s\n",ans2[rep]))
        t1<-gsub(",","+",list_q[j]); t2<-gsub(",","+",matched_question[rep]); t3<-gsub(",","+",ans2[rep]);
        data_out<-rbind(data_out,c(t1,t2,t3))
        cat(sprintf("distance: %f\n",dist[rep]))
        
      }
      
      
      #tf-idf method

      list_dist<-StringMatch(list_q[j],Qmatch,Amatch,drug,weight,"tf")
      lowest_dist<-head(sort(list_dist),2)
      ans_tag<-unlist(sapply(lowest_dist,function(x) which(x==list_dist)))
      ans<-Amatch[ans_tag]
      ans2<-Repmatch[ans_tag]
      matched_question<-Qmatch[ans_tag]
      dist<-list_dist[ans_tag]
      cat(sprintf("method: Tf-idf\n"))
      for (rep in 1:2){
        cat(sprintf("Top answer: %d\n",rep))
        cat(sprintf("matched question: %s\n",matched_question[rep]))
        cat(sprintf("answer: %s\n",ans[rep]))
        t1<-gsub(",","+",list_q[j]); t2<-gsub(",","+",matched_question[rep]); t3<-gsub(",","+",ans[rep]);
        data_out<-rbind(data_out,c(t1,t2,t3))
        cat(sprintf("2nd answer: %s\n",ans2[rep]))
        t1<-gsub(",","+",list_q[j]); t2<-gsub(",","+",matched_question[rep]); t3<-gsub(",","+",ans2[rep]);
        data_out<-rbind(data_out,c(t1,t2,t3))
        cat(sprintf("distance: %f\n",dist[rep]))
        
      }
     
    }
    data_out<-as.data.frame(data_out)
    write.table(data_out,file="data_out.csv",row.names=FALSE,col.names=FALSE,append=TRUE,sep=",")
  } 
  cat(sprintf("true answer: %s\n",answer[index]))
  cat("======================================\n")
}
sink()
# data_out<-as.data.frame(data_out)
# write.table(data_out,file="data_out.csv",row.names=FALSE,col.names=FALSE,sep=",")