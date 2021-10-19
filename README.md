# BERT
## 簡介
[BERT](https://github.com/google-research/bert)

Google目前提供許多版本的開源模型可供使用，可以省下自行花費大量時間及資源去重新訓練BERT模型。
### 什麼是BERT
BERT全名為Bidirectional Encoder Representations from Transformers，顧名思義BERT的架構為Transformers中的Encoder。
<img src="https://i.imgur.com/ZYBeNQF.png" alt="BERT" width="750"/>

BERT 是一種預訓練語言表示的方法，是Google在大型文本語料庫（如維基百科）上訓練一個通用的"語言理解"模型，然後將該模型用於下游NLP任務（如QA）。BERT優於以前的方法，因為它是第一個用於預訓練 NLP 的非監督式深度雙向系統。

非監督式意味著BERT只需要單純的文本資料即可進行訓練，而在網路上隨便都能獲取大量的文本資料。

<br></br>
### Training of BERT
BERT的訓練方式有兩種，這兩種方法是同時使用的。

***Approach1:Masked LM***

Masked LM其實就像是國小寫考卷或習題出現的克漏字測驗，BERT會將input的句子中內 15% 的字隨機以[MASK]替換，BERT模型要預測這個[MASK]是什麼字。
```
Input: the man went to the [MASK1] . he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
```

***Approach2:Next Sentence Prediction***

讓BERT預測兩個句子是否是接在一起的，簡言之就是判斷上下語句的關聯性。 
為了實現這個方法，同樣在input的句子內要加入一些特殊的字，第一個是[SEP]，用來分隔前後句，第二個為[CLS]，代表模型訓練採用Next Senetence Prediction要做分類。
```
Sentence A: the man went to the store .
Sentence B: he bought a gallon of milk .
Label: IsNextSentence
```
```
Sentence A: the man went to the store .
Sentence B: penguins are flightless .
Label: NotNextSentence
```


## Reference
* [李宏毅教授的Youtube](https://www.youtube.com/c/HungyiLeeNTU/videos)
* [Google-Research BERT github](https://github.com/google-research/bert)
