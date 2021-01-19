# Review Analyzer

### Discription
This project is built based on the ABSA task raised in [SemiEval2014](https://alt.qcri.org/semeval2014/task4/). The analyzer will automatically extract aspect terms, analyze the extracted the sentiment of the aspect terms, catogorize the aspect terms into pre-defined categories, and analyzed the sentiment of the aspect categories from the input review. Part of the model is based on the method from (Xu 2019)[1].

### How to Run
python main.py -text "this is where the review goes."</br>

### File
"/model" - this is where you can put ASC, AE and Pretrained models, You can download from [link](https://drive.google.com/file/d/1Q3ALBUAnLA5PcDwHfixwYStao068TTsr/view?usp=sharing).<br>
"/src" - source code</br>
"/src/ABSA/ae" - Aspect Extraction</br>
"/src/AC/asc" - Aspect Sentiment Classifier</br>
"/src/TM" - Topical Modeling source code </br>

### Code Flowchart
![flow chart](https://github.com/wul8/Image/blob/main/minions.png)

### Reference
[1] Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S, BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, 2019


 

