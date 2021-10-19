# BERT
## 簡介
[BERT](https://github.com/google-research/bert)
### 什麼是BERT
BERT全名為Bidirectional Encoder Representations from Transformers，顧名思義BERT的架構為Transformers中的Encoder。
<img src="https://i.imgur.com/ZYBeNQF.png" alt="BERT" width="750"/>

### Training of BERT
BERT的訓練方式有兩種，這兩種方法是同時使用的。

***Approach1:Masked LM***

Masked LM其實就像是國小寫考卷或習題出現的克漏字測驗，BERT會將input的句子中內 15% 的字隨機以[MASK]替換，BERT模型要預測這個[MASK]是什麼字。


***Approach2:Next Sentence Prediction***

讓BERT預測兩個句子是否是接在一起的，簡言之就是判斷上下語句的關聯性。 
為了實現這個方法，同樣在input的句子內要加入一些特殊的字，第一個是[SEP]，用來分隔前後句，第二個為[CLS]，代表模型訓練採用Next Senetence Prediction要做分類。



## Reference
* [李宏毅教授的Youtube](https://www.youtube.com/c/HungyiLeeNTU/videos)
* [Google-Research BERT github](https://github.com/google-research/bert)
