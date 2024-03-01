# EISA
An open-source example of the Episodic Interaction Seperation Architecture (EISA) for episodic memory encodings in LLMs. This examlpe is still under development. 

### Architecture TLDR
Extrapolating from Tulving's theories on sematnic, episodic, and procedural memory, EISA employs a 'separation architecure' which distinguishes forms of episodic memory (i.e. prompter-LLM interaction), from semantic or procedural memory (i.e. pre-trained information or prompter fine-tuning). This approach borrows from the categorization of memory in LLM systems, touched upon in Li & Li 2024 (https://arxiv.org/html/2401.02509v1). We distinguish types of memory via the episodic nessesity checker module (ENCM) which returns a boolean value. Memories are then categorized accordingly. A trainable side network (SideNet) is then used to internalize information about episodic memories which are then cached into a bank specifically for episodic memory which is separate from the bank for fine-tuning information. 

Upon an LLM prompt, the SideNet's iternalized informaiton (a word vector) will be contrasted with relevant information from a given prompt. Episodes are then retrived and fed to the LLM. Every retrival creates a new episodic memory which can then be referenced in future retrivals, allowing for the more complex, chronological organization of episodes. 

# Methods

### The Episodic Nessesity Checker Module (ENCM) 
```
E_I_S_A.ENCM(input_text)
```
The ENCM serves as a classifier for the system to decide whether an input represents a prompt which needs a response that draws from the bank of epiosidc memory (```need=True```) or if the input is a fine tuning ( (```need=False```). ```need``` in this context refers to whether the input needs to be stored as semantic/procedural memory or paired with a response and stored as an episode. 

### Chronological Memory Weights

```
components/mem/mem_arch.py
```
This method combats common issues like memory staleness by encuring that the chronology of memory encodings is recorded. Each episode is attributed a chronology weight which is considered during retrival. The most recent episode will have a weight of $n$ (where $n =$ ```len(self.mem)```) and the most chronologically dated element will have a chronology weight of 1. 

If an episode is retrived, it is then pushed to the top of the chronological list with a weight of $n-1$ just behind the newly formed episode. For example if a prompt, response pair with weight 4 are retrived as well as a pair with weight 6, the new prompt and response are stored with weight $n$, the pair forermly with weight 6 is now stored with weight $n-1$ and the other is stored with weight $n-2$.

### SideNet
```
E_I_S_A.ENCM(input).SideNet(text)
```
The SideNet manages or acts a librarian for the bank of episodic memory. ```if need```, the SideNet will take the prmopt passed by the ENCM, vectorize it, and then check the vecortized words against vectors stored in the episodic memory bank with a cosine similarlty test and account for the chronological memory weights. If vectors are similar by a certain threshold, the SideNet will pass the similar episode to the LLM for generation. It will then store the vectorized format of the prompt just passed by the ENCM along with the response given by the LLM as a new episode in the episodic memory bank. The storage of the  episodes just retrived by the SideNet will also be updated in accordance with the chronological memory weights.
