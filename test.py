# import openai
# import os
# from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("OPEN_AI_API_KEY")

# client = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "what is cognitive science?"}],
#     api_key=API_KEY,
#     stream=False
# )

# print(client.choices[0].message.content)

# str = "what is cognitive science?"
# split = str.split()
# print(len(split))

from transformers import pipeline

# input = "add a meow to every sentence?"

# pipe = pipeline("text-classification", model="krupper/text-complexity-classification")
# res = pipe(input)
# print(res[0]["label"])

# def ENCM(input): 
#     need = None
#     word_c = len(input.split())
#     pipe = pipeline("text-classification", model="krupper/text-complexity-classification")
#     res = pipe(input)
#     label = res[0]["label"]
#     if res[0]["label"] == "special_language":
#         need = True
#     else: 
#         need = False
#     return need

# print(ENCM(input))

from gensim.test.utils import common_texts, common_dictionary, common_corpus
from gensim.models import Word2Vec

# model = Word2Vec(common_texts, min_count=1)


# v_i = model.wv['computer']
# v_j = model.wv['interface']
# print(v_i)

# sim = [-0.01224489 -0.0051019  -0.01572524  0.01780059  0.00604672 -0.00021467
#  -0.00015149  0.00015459  0.00094365  0.00457083 -0.00679132  0.00097008
#  -0.0101938   0.00597107 -0.00689774  0.00359815  0.00151777 -0.0071404
#   0.00341179  0.00150089  0.00353495  0.00699608 -0.01235397  0.00135169
#   0.01469706 -0.00858497 -0.00716308  0.01285604 -0.00903854 -0.00859232
#   0.01247183  0.00351221 -0.00481779  0.00153074 -0.01481628  0.00058753
#  -0.00034591  0.00603093  0.00807043  0.00476542  0.0060518   0.00667689
#   0.00306001 -0.01201011  0.00616438  0.00977315 -0.00050654  0.00757395
#   0.01179073  0.00706729  0.00789534  0.00939301  0.00597889  0.00611901
#   0.00044469 -0.00802493  0.00151431 -0.0007469  -0.00219213 -0.00460817
#   0.00605007  0.01124759  0.00696886 -0.00375304  0.00430409 -0.00927291
#   0.00381078 -0.00221727 -0.00211082 -0.00075332 -0.00103818 -0.00327892
#  -0.00049176 -0.01891911 -0.00016402 -0.0165483  -0.00320925  0.00536804
#   0.01040745 -0.00589026 -0.00204113 -0.01144572 -0.00863741 -0.00038583
#   0.01123121 -0.01014966 -0.00280591 -0.00228018  0.00039861  0.00116556
#  -0.01648971  0.00709532  0.00531241  0.00170201  0.01538601 -0.00732881
#  -0.0184982   0.01131866  0.00557796  0.01162102]

# [ 7.0871473e-03 -1.5683770e-03  7.9461383e-03 -9.4874427e-03
#  -8.0296379e-03 -6.6422881e-03 -4.0041055e-03  4.9910326e-03
#  -3.8136265e-03 -8.3215833e-03  8.4132208e-03 -3.7471859e-03
#   8.6089820e-03 -4.8962692e-03  3.9189286e-03  4.9236105e-03
#   2.3943025e-03 -2.8213640e-03  2.8496366e-03 -8.2571059e-03
#  -2.7652925e-03 -2.5905666e-03  7.2491053e-03 -3.4629775e-03
#  -6.5992209e-03  4.3399399e-03 -4.7540365e-04 -3.5954376e-03
#   6.8824128e-03  3.8715173e-03 -3.8985382e-03  7.7238021e-04
#   9.1438834e-03  7.7564763e-03  6.3607418e-03  4.6693170e-03
#   2.3858559e-03 -1.8414279e-03 -6.3720327e-03 -2.9998214e-04
#  -1.5642056e-03 -5.7058921e-04 -6.2630433e-03  7.4330522e-03
#  -6.5910234e-03 -7.2386782e-03 -2.7575779e-03 -1.5144736e-03
#  -7.6353899e-03  6.9956027e-04 -5.3253169e-03 -1.2739648e-03
#  -7.3665027e-03  1.9612678e-03  3.2734126e-03 -2.4741203e-05
#  -5.4490739e-03 -1.7256952e-03  7.0866030e-03  3.7357591e-03
#  -8.8817989e-03 -3.4116022e-03  2.3567537e-03  2.1376414e-03
#  -9.4648367e-03  4.5697815e-03 -8.6582452e-03 -7.3883547e-03
#   3.4832379e-03 -3.4728223e-03  3.5656220e-03  8.8950340e-03
#  -3.5753327e-03  9.3197329e-03  1.7111654e-03  9.8462272e-03
#   5.7051554e-03 -9.1497730e-03 -3.3269720e-03  6.5306681e-03
#   5.6031067e-03  8.7057864e-03  6.9263605e-03  8.0408510e-03
#  -9.8231174e-03  4.2975070e-03 -5.0308690e-03  3.5132254e-03
#   6.0578999e-03  4.3924116e-03  7.5100451e-03  1.4993458e-03
#  -1.2642510e-03  5.7697673e-03 -5.6368350e-03  3.7107900e-05
#   9.4556073e-03 -5.4809595e-03  3.8159827e-03 -8.1130695e-03]

# [-0.00515774 -0.00667028 -0.0077791   0.00831315 -0.00198292 -0.00685696
#  -0.0041556   0.00514562 -0.00286997 -0.00375075  0.0016219  -0.0027771
#  -0.00158482  0.0010748  -0.00297881  0.00852176  0.00391207 -0.00996176
#   0.00626142 -0.00675622  0.00076966  0.00440552 -0.00510486 -0.00211128
#   0.00809783 -0.00424503 -0.00763848  0.00926061 -0.00215612 -0.00472081
#   0.00857329  0.00428459  0.0043261   0.00928722 -0.00845554  0.00525685
#   0.00203994  0.0041895   0.00169839  0.00446543  0.0044876   0.0061063
#  -0.00320303 -0.00457706 -0.00042664  0.00253447 -0.00326412  0.00605948
#   0.00415534  0.00776685  0.00257002  0.00811905 -0.00138761  0.00808028
#   0.0037181  -0.00804967 -0.00393476 -0.0024726   0.00489447 -0.00087241
#  -0.00283173  0.00783599  0.00932561 -0.0016154  -0.00516075 -0.00470313
#  -0.00484746 -0.00960562  0.00137242 -0.00422615  0.00252744  0.00561612
#  -0.00406709 -0.00959937  0.00154715 -0.00670207  0.0024959  -0.00378173
#   0.00708048  0.00064041  0.00356198 -0.00273993 -0.00171105  0.00765502
#   0.00140809 -0.00585215 -0.00783678  0.00123305  0.00645651  0.00555797
#  -0.00897966  0.00859466  0.00404816  0.00747178  0.00974917 -0.0072917
#  -0.00904259  0.0058377   0.00939395  0.00350795]

input = "what is a human computer interface"
model = Word2Vec(common_texts, min_count=1)
vocab = set(model.wv.index_to_key)
spl = input.split()
filt_mem = [word for word in spl if word in vocab]
        
def SideNet():  
    db = []
    db.extend([model.wv['computer'], model.wv['interface']])
    data = []
    for word in filt_mem:
        vec = model.wv[word]
        data.append([word] + vec.tolist())  
        def Search():
            sim_search = []
            for v in db:
                for j in data: 
                    sim_search.append(abs(v - j))
                for k in sim_search: 
                    final_search = []
                    # if k < 0.5:
                    #     final_search.append(k)
                    #     return final_search 
                    final_search.append(k)
                    return final_search
                    
print(SideNet())