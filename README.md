# EISA
An open-source example of the Episodic Interaction Seperation Architecture (EISA) for episodic memory encodings in LLMs. This examlpe is still under development. 

### System architecure 
Extrapolating from Tulving's theories on sematnic, episodic, and procedural memory, EISA employs a 'separation architecure' which distinguishes forms of episodic memory (i.e. prompter-LLM interaction), from semantic or procedural memory (i.e. pre-trained information or prompter fine-tuning). This approach borrows from the categorization of memory in LLM systems, touched upon in Li & Li 2024 (https://arxiv.org/html/2401.02509v1). We distinguish types of memory via the episodic nessesity checker module (ENCM) which returns a boolean value. Memories are then categorized accordingly. A trainable side network (SideNet) is then used to internalize information about episodic memories which are then cached into a bank specifically for episodic memory which is separate from the bank for fine-tuning information. 

Upon an LLM prompt, the SideNet's iternalized informaiton (a word vector) will be contrasted with relevant information from a given prompt. Episodes are then retrived and fed to the LLM. Every retrival creates a new episodic memory which can then be referenced in future retrivals, allowing for the more complex, chronological organization of episodes. 

### More information is coming soon

### Tentative Peusdo-code Architecure

![Screenshot 2024-02-16 at 10 35 12 PM](https://github.com/shankarchawla1776/EISA/assets/139474458/eec4f80e-3d3e-43e8-85f8-0d5c0967b580)

