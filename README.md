# Finetuning Florence-2
This project aims to give an overview into how to get started using Florence-2, developed by Microsoft.
Florence-2 is a vision foundation model that be used for a variety of computer vision tasks such as:
    
    - Object Detection
    - Captioning
    - Classification
    - Segmentation
    - Optical Character Recognition (OCR)

These notebooks will help you get started with the general useage and limitations as well as show an
example of how to finetune the model for object detection on a simple example. If you have your own data,
`02-Object_Detection_Finetuning.ipynb` will give further instructions how you can modify the notebook to
finetune using your own data.

## Getting started

Clone this repo 

    - `git clone https://github.com/microsoft/dstoolkit-finetuning-florence-2.git`

Once the repo has been cloned, create a new Python enviroment and activate it
    
    - `python -m virtualenv env`
    - `env\Scripts\Activate`

Install Python requirements from requirements.txt
    
    - `pip install -r requirements.txt`

Open and work through `01-Base_Model.ipynb` before `02-Object_Detection_Finetuning.ipynb`.
    
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
