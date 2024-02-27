# EISA
An open-source example of the Episodic Interaction Seperation Architecture (EISA) for episodic memory encodings in LLMs. This examlpe is still under development. 

### System architecure 
Extrapolating from Tulving's theories on sematnic, episodic, and procedural memory, EISA employs a 'separation architecure' which distinguishes forms of episodic memory (i.e. prompter-LLM interaction), from semantic or procedural memory (i.e. pre-trained information or prompter fine-tuning). This approach borrows from the categorization of memory in LLM systems, touched upon in Li & Li 2024 (https://arxiv.org/html/2401.02509v1). We distinguish types of memory via the episodic nessesity checker module (ENCM) which returns a boolean value. Memories are then categorized accordingly. A trainable side network (SideNet) is then used to internalize information about episodic memories which are then cached into a bank specifically for episodic memory which is separate from the bank for fine-tuning information. 

Upon an LLM prompt, the SideNet's iternalized informaiton (a word vector) will be contrasted with relevant information from a given prompt. Episodes are then retrived and fed to the LLM. Every retrival creates a new episodic memory which can then be referenced in future retrivals, allowing for the more complex, chronological organization of episodes. 

# Methods: 

### Chronological Memory Weights

```
components/mem/mem_arch.py
```
This method combats common issues like memory staleness by encuring that the chronology of memory encodings is recorded. Each episode is attributed a chronology weight which is considered during retrival. The most recent episode will have a weight of n (where $n =$ len(self.mem)) and the most chronologically dated element will have a chronology weight of 1. If a memory is retrived, it is then paired with the prompt that issued its retrival and stored at the top of the chronological list with a weight of n. If multiple episodes are retrived in conjunction with a single prompt, the prompt is then stored multiple times, once with each episode it retrives. These are then stored in decending order in line with the chronological weights of the original retrived episodes. For example if a key, value pair with weight 4 are retrived as well as a pair with weight 6, the latter is stored as prompt, (prompt, response) with weight n $ - 1$ and the former is stored after with weight n. 